"""
Demo using pretrained weights.
Run: python examples/demo_pretrained.py
"""

from polyglotlite import PolyglotLiteHF, detect_language

def main():
    print("Loading pretrained model...")
    model = PolyglotLiteHF("polyglot-135m", device="cpu")
    print(model)
    print()
    
    prompts = [
        "The meaning of life is",
        "Machine learning is",
        "Once upon a time",
    ]
    
    print("Generating text:\n")
    for prompt in prompts:
        output = model.generate(prompt, max_length=40, temperature=0.8)
        print(f"Prompt: {prompt}")
        print(f"Output: {output}\n")

if __name__ == "__main__":
    main()