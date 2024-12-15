"""
PolyglotLite Quick Start Example

This example demonstrates the basic usage of PolyglotLite for:
1. Loading a model
2. Generating text
3. Language detection
4. Fine-tuning (basic)

Run with: python examples/quickstart.py
"""

import torch
from polyglotlite import PolyglotLite, detect_language, get_supported_languages


def main():
    print("=" * 60)
    print("PolyglotLite Quick Start")
    print("=" * 60)
    
    # 1. Check available languages
    print("\n1. Supported Languages")
    print("-" * 40)
    languages = get_supported_languages()
    print(f"Total supported languages: {len(languages)}")
    sample_langs = list(languages.items())[:10]
    for code, name in sample_langs:
        print(f"  {code}: {name}")
    print("  ...")
    
    # 2. Language detection
    print("\n2. Language Detection")
    print("-" * 40)
    
    test_texts = [
        ("Hello, how are you today?", "English"),
        ("Bonjour, comment allez-vous?", "French"),
        ("Hola, ¿cómo estás?", "Spanish"),
        ("Guten Tag, wie geht es Ihnen?", "German"),
        ("你好，你好吗？", "Chinese"),
        ("こんにちは、お元気ですか？", "Japanese"),
        ("مرحبا، كيف حالك؟", "Arabic"),
        ("Привет, как дела?", "Russian"),
    ]
    
    for text, expected in test_texts:
        detected = detect_language(text)
        lang_name = languages.get(detected, "Unknown")
        status = "✓" if lang_name.lower().startswith(expected.lower()[:3]) else "?"
        print(f"  {status} '{text[:30]}...' -> {detected} ({lang_name})")
    
    # 3. Initialize model
    print("\n3. Model Initialization")
    print("-" * 40)
    
    # Model defaults to CPU for stability on Apple Silicon
    model = PolyglotLite(model_name="polyglot-135m")
    print(f"Device: {model.device}")
    print(model)
    
    # 4. Text generation (basic demo)
    print("\n4. Text Generation Demo")
    print("-" * 40)
    
    model.load_tokenizer()
    
    prompts = [
        "The future of artificial intelligence is",
        "Machine learning can help us",
        "Hello, my name is",
    ]
    
    print("Generating text (this may take a moment on first run)...\n")
    
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        # Short generation for demo
        output = model.generate(
            prompt, 
            max_length=20,
            temperature=0.7,
            do_sample=True
        )
        print(f"Output: {output}\n")
    
    # 5. Model info
    print("\n5. Model Information")
    print("-" * 40)
    
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 2 / 1e6:.1f} MB (FP16)")
    print(f"  Device: {model.device}")
    
    print("\n" + "=" * 60)
    print("Quick start complete! See README.md for more examples.")
    print("=" * 60)


if __name__ == "__main__":
    main()
