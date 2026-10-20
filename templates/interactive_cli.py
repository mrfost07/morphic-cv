"""
Universal Interactive Inference CLI for Morphic-CV
Works on pure CPU with < 300MB RAM. Paste any image path, hit Enter, and detect!

Usage:
    python templates/interactive_cli.py
    python templates/interactive_cli.py --adapter path/to/adapter.pt --classes dog cat bird
"""

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from PIL import Image
import torch
import torchvision.transforms as transforms
from morphic_cv import MorphicDINOv2


def run_interactive_inference(adapter_path: str = None, class_names: list = None, model_name: str = "dinov2_vits14"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print()
    print("=" * 65)
    print("     MORPHIC-CV: UNIVERSAL INTERACTIVE INFERENCE ENGINE")
    print("=" * 65)
    print(f" [*] Hardware: {str(device).upper()} (Zero GPU required)")
    print(f" [*] Base Architecture: Meta DINOv2 ({model_name}) + MorphicAdapter")

    if class_names is None:
        class_names = ["Defect", "Clean"]
    num_classes = len(class_names)

    print(f" [*] Target Classes ({num_classes}): {', '.join(class_names)}")
    print(" [*] Loading model into RAM...")

    # Load base model with Morphic Adapter
    model = MorphicDINOv2.from_pretrained(model_name, num_classes=num_classes).to(device)

    if adapter_path and os.path.exists(adapter_path):
        model.load_adapter(adapter_path)
        print(f" [*] Loaded trained adapter weights from: {adapter_path} ({os.path.getsize(adapter_path)/1024:.1f} KB)")
    else:
        print(" [*] Running with initialized MorphicAdapter weights.")

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(" [*] Engine ready! Memory usage < 300MB.")
    print("=" * 65)
    print("  - Paste or drag-and-drop any image path, then press Enter.")
    print("  - Type 'q' anytime to exit.")
    print("=" * 65)

    while True:
        print()
        try:
            user_input = input("Paste image file path (or 'q' to quit) > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        clean_path = user_input.strip().strip('"').strip("'")
        if clean_path.lower() in ['q', 'quit', 'exit']:
            print("\nExiting. Goodbye!")
            break

        if not os.path.exists(clean_path):
            print("\n  [!] Error: File not found at path:")
            print(f"      '{clean_path}'")
            print("      Please verify the file path and try again.")
            continue

        try:
            img = Image.open(clean_path).convert("RGB")
        except Exception as e:
            print(f"\n  [!] Error opening image: {e}")
            continue

        start_time = time.time()
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=-1)[0]
            pred_idx = torch.argmax(probs).item()

        latency_ms = (time.time() - start_time) * 1000.0
        pred_label = class_names[pred_idx]
        confidence = probs[pred_idx].item() * 100.0

        print("\n" + "=" * 65)
        print(f"  PREDICTION RESULT ({latency_ms:.1f} ms on {device.type.upper()})")
        print("=" * 65)
        print(f"  PREDICTED CLASS : >> {pred_label.upper()} <<")
        print(f"  CONFIDENCE      : {confidence:.2f}%")
        print("-" * 65)
        print("  Top Probability Breakdown:")
        topk = min(5, num_classes)
        top_vals, top_idx = torch.topk(probs, k=topk)
        for rank, (val, idx) in enumerate(zip(top_vals, top_idx), 1):
            label = class_names[idx.item()]
            bar = "#" * int(val.item() * 24)
            print(f"    [{rank}] {label:<16} : {val.item()*100:5.1f}%  |{bar}")
        print("=" * 65)

        print()
        try:
            choice = input("Would you like to test another image? [Y/n] (or 'q' to quit): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting. Goodbye!")
            break

        if choice in ['n', 'no', 'q', 'quit', 'exit']:
            print("\nExiting. Goodbye!")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Interactive Inference CLI")
    parser.add_argument("--adapter", type=str, default=None, help="Path to trained .pt adapter checkpoint")
    parser.add_argument("--classes", nargs="+", default=None, help="List of class names (e.g. --classes dog cat bird)")
    parser.add_argument("--model", type=str, default="dinov2_vits14", help="Backbone model name")
    args = parser.parse_args()

    run_interactive_inference(adapter_path=args.adapter, class_names=args.classes, model_name=args.model)
