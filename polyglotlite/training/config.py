"""Training configuration for PolyglotLite"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    Config for training. Most defaults are reasonable for a single GPU.
    Adjust batch_size and gradient_accumulation_steps based on your memory.
    """
    
    # Model architecture
    model_size: str = "135m"
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 4  # for grouped-query attention
    intermediate_size: int = 3072
    max_seq_len: int = 2048
    dropout: float = 0.0
    
    # Training hyperparameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # Optimization
    use_amp: bool = True  # Automatic mixed precision
    use_gradient_checkpointing: bool = False
    use_flash_attention: bool = False
    
    # Fine-tuning
    use_lora: bool = False
    use_qlora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    # Data
    languages: List[str] = field(default_factory=lambda: ["en"])
    dataset: str = "wikipedia"
    data_path: Optional[str] = None
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    output_dir: str = "./outputs"
    wandb_project: Optional[str] = None
    
    # Hardware
    device: str = "auto"
    num_workers: int = 4
    
    def __post_init__(self):
        """Set model-specific defaults based on model_size."""
        model_configs = {
            "135m": {
                "hidden_size": 768,
                "num_layers": 12,
                "num_heads": 12,
                "intermediate_size": 3072,
            },
            "360m": {
                "hidden_size": 1024,
                "num_layers": 24,
                "num_heads": 16,
                "intermediate_size": 4096,
            },
            "500m": {
                "hidden_size": 1280,
                "num_layers": 28,
                "num_heads": 20,
                "intermediate_size": 5120,
            },
        }
        
        if self.model_size in model_configs:
            for key, value in model_configs[self.model_size].items():
                if getattr(self, key) == getattr(TrainingConfig, key, None):
                    setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
    
    def __repr__(self) -> str:
        return f"TrainingConfig(model_size={self.model_size}, batch_size={self.batch_size})"


@dataclass  
class LoRAConfig:
    """Configuration for LoRA/QLoRA fine-tuning."""
    
    r: int = 8  # LoRA rank
    alpha: int = 16  # LoRA alpha
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: str = "none"  # "none", "all", or "lora_only"
    
    # QLoRA specific
    use_4bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_optimal_config(
    available_memory_gb: float,
    target_languages: List[str],
    use_qlora: bool = False
) -> TrainingConfig:
    """
    Get optimal training configuration based on available resources.
    
    Args:
        available_memory_gb: Available GPU/system memory in GB
        target_languages: List of target language codes
        use_qlora: Whether to use QLoRA for memory efficiency
        
    Returns:
        Optimized TrainingConfig
    """
    config = TrainingConfig()
    config.languages = target_languages
    
    if available_memory_gb < 8:
        # Very limited memory (e.g., free Colab)
        config.model_size = "135m"
        config.batch_size = 1
        config.gradient_accumulation_steps = 32
        config.max_seq_len = 512
        config.use_gradient_checkpointing = True
        config.use_qlora = True
    elif available_memory_gb < 16:
        # Moderate memory (e.g., gaming laptop)
        config.model_size = "135m"
        config.batch_size = 4
        config.gradient_accumulation_steps = 8
        config.max_seq_len = 1024
    elif available_memory_gb < 24:
        # Good memory (e.g., M4 Pro)
        config.model_size = "360m"
        config.batch_size = 4
        config.gradient_accumulation_steps = 8
        config.max_seq_len = 1024
    else:
        # High memory (e.g., A100)
        config.model_size = "500m"
        config.batch_size = 16
        config.gradient_accumulation_steps = 2
        config.max_seq_len = 2048
        config.use_flash_attention = True
    
    if use_qlora:
        config.use_qlora = True
        config.batch_size = max(1, config.batch_size // 2)
    
    return config
