"""
Tests for PolyglotLite

Run with: pytest tests/
"""

import pytest
import torch

from polyglotlite import PolyglotLite, TrainingConfig
from polyglotlite.tokenizers import SimpleTokenizer
from polyglotlite.utils import detect_language, get_supported_languages


class TestPolyglotLiteModel:
    """Tests for the PolyglotLite model."""
    
    def test_model_initialization(self):
        """Test that model can be initialized."""
        model = PolyglotLite(model_name="polyglot-135m")
        assert model is not None
        assert model.config["hidden_size"] == 768
        assert model.config["num_layers"] == 12
    
    def test_model_forward_pass(self):
        """Test forward pass with dummy input."""
        model = PolyglotLite(model_name="polyglot-135m")
        model.model.eval()
        
        # Create dummy input
        input_ids = torch.randint(0, 1000, (1, 32))
        
        with torch.no_grad():
            output = model.model(input_ids)
        
        assert "logits" in output
        assert output["logits"].shape == (1, 32, model.config["vocab_size"])
    
    def test_model_parameter_count(self):
        """Test that parameter count matches expected."""
        model = PolyglotLite(model_name="polyglot-135m")
        total_params = sum(p.numel() for p in model.model.parameters())
        
        # Should be around 135M (allowing some variance)
        assert 100_000_000 < total_params < 200_000_000
    
    def test_model_device_auto(self):
        """Test automatic device selection."""
        model = PolyglotLite(model_name="polyglot-135m")
        # Should default to CPU in test environment
        assert model.device in ["cpu", "cuda", "mps"]


class TestTokenizer:
    """Tests for the tokenizer."""
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initialization."""
        tokenizer = SimpleTokenizer(vocab_size=32000)
        assert len(tokenizer) > 0
    
    def test_encode_decode_roundtrip(self):
        """Test that encoding then decoding returns original text."""
        tokenizer = SimpleTokenizer(vocab_size=32000)
        
        text = "Hello, world!"
        tokens = tokenizer.encode(text, add_bos=False, add_eos=False)
        decoded = tokenizer.decode(tokens)
        
        assert decoded == text
    
    def test_special_tokens(self):
        """Test special token IDs."""
        tokenizer = SimpleTokenizer(vocab_size=32000)
        
        assert tokenizer.pad_token_id == 0
        assert tokenizer.unk_token_id == 1
        assert tokenizer.bos_token_id == 2
        assert tokenizer.eos_token_id == 3
    
    def test_multilingual_encoding(self):
        """Test encoding of multilingual text."""
        tokenizer = SimpleTokenizer(vocab_size=32000)
        
        texts = [
            "Hello",  # English
            "Bonjour",  # French
            "Hola",  # Spanish
            "你好",  # Chinese
            "こんにちは",  # Japanese
        ]
        
        for text in texts:
            tokens = tokenizer.encode(text)
            assert len(tokens) > 0
            decoded = tokenizer.decode(tokens)
            # At minimum, we should get something back
            assert len(decoded) > 0


class TestLanguageDetection:
    """Tests for language detection."""
    
    def test_english_detection(self):
        """Test detection of English text."""
        text = "The quick brown fox jumps over the lazy dog."
        lang = detect_language(text)
        assert lang == "en"
    
    def test_spanish_detection(self):
        """Test detection of Spanish text."""
        text = "El rápido zorro marrón salta sobre el perro perezoso."
        lang = detect_language(text)
        assert lang == "es"
    
    def test_french_detection(self):
        """Test detection of French text."""
        text = "Le renard brun rapide saute par-dessus le chien paresseux."
        lang = detect_language(text)
        assert lang == "fr"
    
    def test_chinese_detection(self):
        """Test detection of Chinese text."""
        text = "敏捷的棕色狐狸跳过了懒狗。"
        lang = detect_language(text)
        assert lang == "zh"
    
    def test_japanese_detection(self):
        """Test detection of Japanese text."""
        text = "すばやい茶色のキツネはのろまな犬を飛び越える。"
        lang = detect_language(text)
        assert lang == "ja"
    
    def test_arabic_detection(self):
        """Test detection of Arabic text."""
        text = "الثعلب البني السريع يقفز فوق الكلب الكسول"
        lang = detect_language(text)
        assert lang == "ar"
    
    def test_supported_languages(self):
        """Test that supported languages are returned."""
        languages = get_supported_languages()
        assert "en" in languages
        assert "es" in languages
        assert "zh" in languages
        assert len(languages) > 50


class TestTrainingConfig:
    """Tests for training configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        
        assert config.model_size == "135m"
        assert config.batch_size == 8
        assert config.learning_rate == 5e-4
    
    def test_model_size_presets(self):
        """Test model size presets are applied."""
        config_135m = TrainingConfig(model_size="135m")
        config_360m = TrainingConfig(model_size="360m")
        
        assert config_135m.hidden_size == 768
        assert config_360m.hidden_size == 1024
    
    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = TrainingConfig(batch_size=16)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["batch_size"] == 16


class TestGeneration:
    """Tests for text generation (basic smoke tests)."""
    
    def test_generate_basic(self):
        """Test basic text generation."""
        model = PolyglotLite(model_name="polyglot-135m")
        model.load_tokenizer()
        
        # Generate with very short output for speed
        output = model.generate("Hello", max_length=5, do_sample=False)
        
        assert isinstance(output, str)
        assert len(output) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
