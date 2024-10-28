# PolyglotLite

**Lightweight multilingual language models that run on consumer hardware.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PolyglotLite is a toolkit for training and running efficient multilingual language models (100M-500M parameters) without needing expensive cloud GPUs. It's designed for developers who want to experiment with multilingual NLP on their own machines.

## Status

This is an early-stage project. The model architecture is implemented but pretrained weights are still in development. You can:
- Train models from scratch on your own data
- Fine-tune on custom datasets  
- Experiment with the architecture
- Use the language detection utilities

Pretrained weights coming soon.

## Quick Start

### Installation

```bash
git clone https://github.com/IIIDman/polyglotlite.git
cd polyglotlite
pip install -e .
```

### Basic Usage

```python
from polyglotlite import PolyglotLite

# Initialize a model
model = PolyglotLite(model_name="polyglot-135m")

# Generate text (note: without pretrained weights, output will be random)
output = model.generate("Hello world", max_length=50)
```

### Training on Your Data

```python
from polyglotlite import PolyglotLite, Trainer

model = PolyglotLite(model_name="polyglot-135m")

trainer = Trainer(
    model=model,
    train_data="path/to/your/data.json",
    learning_rate=2e-4,
    batch_size=8
)
trainer.train()

model.save_pretrained("my-model")
```

### Language Detection

```python
from polyglotlite import detect_language

detect_language("Bonjour le monde")  # returns 'fr'
detect_language("你好世界")  # returns 'zh'
```

## Model Sizes

| Model | Parameters | Memory (FP16) | 
|-------|------------|---------------|
| polyglot-135m | 135M | ~270MB |
| polyglot-360m | 360M | ~720MB |
| polyglot-500m | 500M | ~1GB |

## Supported Languages

The tokenizer and language detection support 50+ languages including English, Chinese, Spanish, French, German, Portuguese, Russian, Japanese, Korean, Arabic, Hindi, Vietnamese, Turkish, Polish, and many others. See `polyglotlite/utils/language.py` for the full list.

## Compatibility

**Platforms:** macOS (Intel & Apple Silicon), Linux, Windows

**Python:** 3.8 - 3.13

**Apple Silicon Note:** For stability on M1/M2/M3/M4 Macs, the model defaults to CPU. You can try MPS acceleration with `device="mps"` but it may have issues with some PyTorch operations.

## Project Structure

```
polyglotlite/
├── polyglotlite/
│   ├── models/          # Model architecture
│   ├── tokenizers/      # Tokenization  
│   ├── training/        # Training loop, configs
│   ├── inference/       # (planned) Optimized inference
│   └── utils/           # Language detection, helpers
├── examples/            
├── tests/               
└── scripts/             
```

## Troubleshooting

**MPS errors on Mac:**
```python
model = PolyglotLite.from_pretrained("polyglot-135m", device="cpu")
```

**Import errors:** Make sure you're in the directory containing `pyproject.toml` when running `pip install -e .`

## License

MIT
