"""
Core PolyglotLite Model Implementation

A lightweight, efficient multilingual language model architecture
optimized for consumer hardware.
"""

import math
import json
import os
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm - simpler and faster than LayerNorm"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x: torch.Tensor, seq_len: int) -> tuple:
        # Create position indices on the same device as input
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple:
    """Apply rotary positional embeddings to queries and keys."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with Grouped Query Attention (GQA) support."""
    
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_size // num_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple] = None,
    ) -> tuple:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(q, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))
        
        # Handle KV cache for generation
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        
        past_key_value = (k, v)
        
        # Repeat KV heads for GQA
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply causal mask
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(seq_len, k.shape[2], dtype=torch.bool, device=q.device),
                diagonal=k.shape[2] - seq_len + 1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        else:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.o_proj(attn_output), past_key_value


class FeedForward(nn.Module):
    """FFN with SwiGLU activation (GLU variant that seems to work best)"""
    
    def __init__(self, hidden_size: int, intermediate_size: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        intermediate_size = intermediate_size or hidden_size * 4
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: swish(gate) * up, then project down
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class TransformerBlock(nn.Module):
    """Single transformer decoder block."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, num_kv_heads, dropout)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout)
        self.attention_norm = RMSNorm(hidden_size)
        self.ffn_norm = RMSNorm(hidden_size)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple] = None,
    ) -> tuple:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states, past_key_value = self.attention(hidden_states, attention_mask, past_key_value)
        hidden_states = residual + hidden_states
        
        # Feed-forward with residual
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, past_key_value


class PolyglotLiteModel(nn.Module):
    """Core PolyglotLite transformer model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=config["hidden_size"],
                num_heads=config["num_heads"],
                num_kv_heads=config.get("num_kv_heads"),
                intermediate_size=config.get("intermediate_size"),
                dropout=config.get("dropout", 0.0),
            )
            for _ in range(config["num_layers"])
        ])
        self.norm = RMSNorm(config["hidden_size"])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[tuple]] = None,
        use_cache: bool = False,
    ) -> Dict[str, Any]:
        hidden_states = self.embed_tokens(input_ids)
        
        new_past_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, new_past_kv = layer(hidden_states, attention_mask, past_kv)
            if use_cache:
                new_past_key_values.append(new_past_kv)
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return {
            "logits": logits,
            "past_key_values": new_past_key_values,
        }


class PolyglotLite:
    """
    Main interface for PolyglotLite models.
    
    Example:
        >>> model = PolyglotLite.from_pretrained("polyglot-135m")
        >>> output = model.generate("Hello, world!", max_length=50)
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        "polyglot-135m": {
            "vocab_size": 32000,
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "num_kv_heads": 4,
            "intermediate_size": 3072,
            "max_seq_len": 2048,
            "dropout": 0.0,
        },
        "polyglot-360m": {
            "vocab_size": 32000,
            "hidden_size": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "num_kv_heads": 4,
            "intermediate_size": 4096,
            "max_seq_len": 2048,
            "dropout": 0.0,
        },
        "polyglot-500m": {
            "vocab_size": 32000,
            "hidden_size": 1280,
            "num_layers": 28,
            "num_heads": 20,
            "num_kv_heads": 4,
            "intermediate_size": 5120,
            "max_seq_len": 2048,
            "dropout": 0.0,
        },
    }
    
    def __init__(
        self, 
        config: Optional[Union[Dict, "TrainingConfig"]] = None,
        model_name: str = "polyglot-135m"
    ):
        """
        Initialize PolyglotLite model.
        
        Args:
            config: Model configuration dict or TrainingConfig
            model_name: Name of predefined model config
        """
        if config is None:
            config = self.MODEL_CONFIGS.get(model_name, self.MODEL_CONFIGS["polyglot-135m"])
        elif hasattr(config, "__dict__"):
            config = config.__dict__
            
        self.config = config
        self.model = PolyglotLiteModel(config)
        self.tokenizer = None
        # Default to CPU - don't auto-detect to avoid MPS issues
        self.device = "cpu"
    
    def to(self, device: str) -> "PolyglotLite":
        """Move model to specified device."""
        self.device = device
        self.model = self.model.to(device)
        return self
    
    def load_tokenizer(self, tokenizer_path: Optional[str] = None):
        """Load tokenizer from file or use default."""
        from polyglotlite.tokenizers.tokenizer import SimpleTokenizer
        self.tokenizer = SimpleTokenizer(vocab_size=self.config["vocab_size"])
    
    @classmethod
    def from_pretrained(
        cls, 
        model_name_or_path: str, 
        device: str = "auto",
        **kwargs
    ) -> "PolyglotLite":
        """
        Load a pretrained model.
        
        Args:
            model_name_or_path: Model name or path to saved model
            device: Device to load model on ("auto", "cpu", "cuda", "mps")
            
        Returns:
            PolyglotLite instance
        """
        # Check if it's a predefined model
        if model_name_or_path in cls.MODEL_CONFIGS:
            config = cls.MODEL_CONFIGS[model_name_or_path]
        else:
            # Try to load from path
            config_path = Path(model_name_or_path) / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
            else:
                raise ValueError(f"Unknown model: {model_name_or_path}")
        
        instance = cls(config=config)
        
        # Load weights if available
        weights_path = Path(model_name_or_path) / "model.pt"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu")
            instance.model.load_state_dict(state_dict)
        
        # Set device - default to CPU for stability
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                # Default to CPU even if MPS is available (MPS can be unstable)
                device = "cpu"
        
        instance.to(device)
        instance.load_tokenizer()
        
        return instance
    
    def save_pretrained(self, save_path: str):
        """
        Save model to directory.
        
        Args:
            save_path: Directory to save model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(save_path / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)
        
        # Save model weights
        torch.save(self.model.state_dict(), save_path / "model.pt")
        
        print(f"Model saved to {save_path}")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling (vs greedy decoding)
            
        Returns:
            Generated text
        """
        if self.tokenizer is None:
            self.load_tokenizer()
        
        self.model.eval()
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        generated = input_ids.tolist()[0]
        past_key_values = None
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model output
                if past_key_values is None:
                    curr_input = input_ids
                else:
                    curr_input = torch.tensor([[generated[-1]]], dtype=torch.long, device=self.device)
                
                outputs = self.model(curr_input, past_key_values=past_key_values, use_cache=True)
                logits = outputs["logits"][:, -1, :]
                past_key_values = outputs["past_key_values"]
                
                # Apply temperature
                if temperature > 0:
                    logits = logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                # Sample or greedy
                probs = F.softmax(logits, dim=-1)
                if do_sample:
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    next_token = torch.argmax(probs, dim=-1)
                
                generated.append(next_token.item())
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        return self.tokenizer.decode(generated)
    
    def quantize(self, bits: int = 8) -> "PolyglotLite":
        """
        Quantize model for efficient deployment.
        
        Args:
            bits: Quantization bits (4 or 8)
            
        Returns:
            Self for chaining
        """
        if bits == 8:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear}, dtype=torch.qint8
            )
        elif bits == 4:
            print("4-bit quantization requires bitsandbytes library")
        else:
            raise ValueError(f"Unsupported quantization bits: {bits}")
        
        return self
    
    def export_onnx(self, output_path: str, opset_version: int = 14):
        """
        Export model to ONNX format for deployment.
        
        Args:
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
        """
        dummy_input = torch.randint(0, 1000, (1, 32), device=self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=opset_version,
        )
        print(f"Model exported to {output_path}")
    
    def __repr__(self) -> str:
        total_params = sum(p.numel() for p in self.model.parameters())
        return (
            f"PolyglotLite(\n"
            f"  parameters={total_params:,},\n"
            f"  hidden_size={self.config['hidden_size']},\n"
            f"  num_layers={self.config['num_layers']},\n"
            f"  num_heads={self.config['num_heads']},\n"
            f"  device={self.device}\n"
            f")"
        )

class PolyglotLiteHF:
    """
    PolyglotLite using HuggingFace pretrained models as backend.
    Use this for inference with real pretrained weights.
    """
    
    PRETRAINED_MODELS = {
        "polyglot-135m": "HuggingFaceTB/SmolLM-135M",
        "polyglot-360m": "HuggingFaceTB/SmolLM-360M", 
        "polyglot-500m": "Qwen/Qwen2-0.5B",
    }
    
    def __init__(self, model_name: str = "polyglot-135m", device: str = "auto"):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")
        
        hf_model_name = self.PRETRAINED_MODELS.get(model_name, model_name)
        
        # Auto device selection
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.model_name = model_name
        
        print(f"Loading {hf_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            torch_dtype=torch.float32,
            device_map=None
        ).to(device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded on {device}")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def __repr__(self) -> str:
        params = sum(p.numel() for p in self.model.parameters())
        return f"PolyglotLiteHF(model={self.model_name}, parameters={params:,}, device={self.device})"
