"""
Multilingual Tokenizer for PolyglotLite
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
from collections import defaultdict


class SimpleTokenizer:
    """
    Simple character/word-level tokenizer for MVP.
    
    TODO: Replace with SentencePiece for production use.
    This is fine for testing but won't give great results on real text.
    """
    
    # Special tokens
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    
    def __init__(
        self, 
        vocab_size: int = 32000,
        vocab_file: Optional[str] = None
    ):
        """
        Initialize tokenizer.
        
        Args:
            vocab_size: Size of vocabulary
            vocab_file: Path to vocabulary file (optional)
        """
        self.vocab_size = vocab_size
        
        # Initialize special tokens
        self.special_tokens = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.BOS_TOKEN: 2,
            self.EOS_TOKEN: 3,
        }
        
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        if vocab_file and Path(vocab_file).exists():
            self.load_vocab(vocab_file)
        else:
            self._build_default_vocab()
    
    def _build_default_vocab(self):
        """Build a default vocabulary with common characters."""
        self.token_to_id = dict(self.special_tokens)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        # Add ASCII printable characters
        current_id = len(self.special_tokens)
        for i in range(32, 127):
            char = chr(i)
            if char not in self.token_to_id:
                self.token_to_id[char] = current_id
                self.id_to_token[current_id] = char
                current_id += 1
        
        # Add common Unicode ranges for multilingual support
        unicode_ranges = [
            (0x00C0, 0x00FF),  # Latin Extended-A
            (0x0100, 0x017F),  # Latin Extended-B
            (0x0400, 0x04FF),  # Cyrillic
            (0x0600, 0x06FF),  # Arabic
            (0x0900, 0x097F),  # Devanagari (Hindi)
            (0x4E00, 0x4FFF),  # CJK (subset for demo)
            (0x3040, 0x309F),  # Hiragana
            (0x30A0, 0x30FF),  # Katakana
            (0xAC00, 0xAD00),  # Korean (subset)
        ]
        
        for start, end in unicode_ranges:
            for i in range(start, min(end + 1, start + 200)):  # Limit for demo
                if current_id >= self.vocab_size:
                    break
                char = chr(i)
                if char not in self.token_to_id:
                    self.token_to_id[char] = current_id
                    self.id_to_token[current_id] = char
                    current_id += 1
        
        # Fill remaining vocab with byte-level fallback
        for i in range(256):
            if current_id >= self.vocab_size:
                break
            byte_token = f"<0x{i:02X}>"
            if byte_token not in self.token_to_id:
                self.token_to_id[byte_token] = current_id
                self.id_to_token[current_id] = byte_token
                current_id += 1
    
    def encode(
        self, 
        text: str, 
        add_bos: bool = True,
        add_eos: bool = False,
        max_length: Optional[int] = None
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_bos: Add beginning-of-sequence token
            add_eos: Add end-of-sequence token
            max_length: Maximum sequence length
            
        Returns:
            List of token IDs
        """
        tokens = []
        
        if add_bos:
            tokens.append(self.bos_token_id)
        
        # Character-level tokenization with fallback
        for char in text:
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                # Byte-level fallback
                for byte in char.encode('utf-8'):
                    byte_token = f"<0x{byte:02X}>"
                    if byte_token in self.token_to_id:
                        tokens.append(self.token_to_id[byte_token])
                    else:
                        tokens.append(self.unk_token_id)
        
        if add_eos:
            tokens.append(self.eos_token_id)
        
        if max_length:
            tokens = tokens[:max_length]
        
        return tokens
    
    def decode(
        self, 
        token_ids: List[int], 
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output
            
        Returns:
            Decoded text
        """
        special_ids = set(self.special_tokens.values()) if skip_special_tokens else set()
        
        chars = []
        byte_buffer = []
        
        for token_id in token_ids:
            if token_id in special_ids:
                continue
            
            if token_id not in self.id_to_token:
                continue
            
            token = self.id_to_token[token_id]
            
            # Handle byte tokens
            if token.startswith("<0x") and token.endswith(">"):
                byte_val = int(token[3:5], 16)
                byte_buffer.append(byte_val)
            else:
                # Flush byte buffer
                if byte_buffer:
                    try:
                        chars.append(bytes(byte_buffer).decode('utf-8'))
                    except:
                        pass
                    byte_buffer = []
                chars.append(token)
        
        # Flush remaining bytes
        if byte_buffer:
            try:
                chars.append(bytes(byte_buffer).decode('utf-8'))
            except:
                pass
        
        return ''.join(chars)
    
    def save_vocab(self, path: str):
        """Save vocabulary to file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab_size': self.vocab_size,
                'token_to_id': self.token_to_id,
            }, f, ensure_ascii=False, indent=2)
    
    def load_vocab(self, path: str):
        """Load vocabulary from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab_size = data['vocab_size']
        self.token_to_id = data['token_to_id']
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
    
    def __len__(self) -> int:
        return len(self.token_to_id)
    
    def __repr__(self) -> str:
        return f"SimpleTokenizer(vocab_size={len(self)})"


class SentencePieceTokenizer:
    """
    SentencePiece tokenizer wrapper for production use.
    
    Requires sentencepiece library to be installed.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize SentencePiece tokenizer.
        
        Args:
            model_path: Path to .model file
        """
        try:
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(model_path)
        except ImportError:
            raise ImportError(
                "sentencepiece is required for SentencePieceTokenizer. "
                "Install with: pip install sentencepiece"
            )
        
        self.pad_token_id = self.sp.pad_id()
        self.unk_token_id = self.sp.unk_id()
        self.bos_token_id = self.sp.bos_id()
        self.eos_token_id = self.sp.eos_id()
    
    def encode(
        self, 
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
        max_length: Optional[int] = None
    ) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.sp.Encode(text, add_bos=add_bos, add_eos=add_eos)
        if max_length:
            tokens = tokens[:max_length]
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        return self.sp.Decode(token_ids)
    
    def __len__(self) -> int:
        return self.sp.GetPieceSize()


def train_sentencepiece(
    input_file: str,
    output_prefix: str,
    vocab_size: int = 32000,
    model_type: str = "bpe",
    character_coverage: float = 0.9995,
):
    """
    Train a SentencePiece tokenizer.
    
    Args:
        input_file: Path to training text file
        output_prefix: Prefix for output files
        vocab_size: Target vocabulary size
        model_type: "bpe" or "unigram"
        character_coverage: Character coverage for training
    """
    try:
        import sentencepiece as spm
    except ImportError:
        raise ImportError(
            "sentencepiece is required. Install with: pip install sentencepiece"
        )
    
    spm.SentencePieceTrainer.Train(
        input=input_file,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )
    
    print(f"Tokenizer saved to {output_prefix}.model")
