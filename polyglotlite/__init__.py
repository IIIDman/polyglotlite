"""
PolyglotLite: Efficient Multilingual Language Models for Everyone

A toolkit for training, fine-tuning, and deploying efficient multilingual
language models (100M-500M parameters) on consumer hardware.
"""

__version__ = "0.2.0"
__author__ = "Dmitriy Tsarev"

from polyglotlite.models.polyglot import PolyglotLite, PolyglotLiteHF
from polyglotlite.training.trainer import Trainer
from polyglotlite.training.config import TrainingConfig
from polyglotlite.utils.language import detect_language, get_supported_languages

__all__ = [
    "PolyglotLite",
    "PolyglotLiteHF",
    "Trainer", 
    "TrainingConfig",
    "detect_language",
    "get_supported_languages",
]
