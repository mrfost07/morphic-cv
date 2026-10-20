"""
Interactive Room Recognition Script:
Paste or drag-and-drop ANY building image path, hit Enter, and detect the room!
Supports continuous testing: choose whether to test another image or press 'q' to quit.

Usage:
   python predict_room.py
   or double-click run_detector.bat
"""

import os
import sys
import time
import warnings

# Suppress noisy library warnings for a clean CMD experience
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from morphic_cv.adapters.msa import MorphicAdapter


MODEL_FILE = "thesis_room_model.pt"
CACHE_FILE = "building_features_cache.pt"

# Default list of all 46 rooms (in sorted order matching training)
DEFAULT_CLASSES = [
    '2nd Floor_201', '2nd Floor_202', '2nd Floor_203', '2nd Floor_204', '2nd Floor_205',
    '2nd Floor_206', '2nd Floor_207', '2nd Floor_208', '2nd Floor_209', '2nd Floor_210',
    '3rd Floor_301', '3rd Floor_302', '3rd Floor_303', '3rd Floor_304', '3rd Floor_305',
    '3rd Floor_306', '3rd Floor_307', '3rd Floor_308', '3rd Floor_309', '3rd Floor_310',
    '3rd Floor_311', '3rd Floor_312',
    '4th Floor_401', '4th Floor_402', '4th Floor_403', '4th Floor_404', '4th Floor_405',
    '4th Floor_406', '4th Floor_407', '4th Floor_408', '4th Floor_409', '4th Floor_410',
    '4th Floor_411', '4th Floor_412',
    '5th Floor_501', '5th Floor_502', '5th Floor_503', '5th Floor_504', '5th Floor_505',
    '5th Floor_506', '5th Floor_507', '5th Floor_508', '5th Floor_509', '5th Floor_510',
    '5th Floor_511', '5th Floor_512'
]


class MorphicRoomClassifier(nn.Module):
    def __init__(self, embed_dim=384, bottleneck=32, num_classes=46):
        super().__init__()
        self.adapter = MorphicAdapter(in_channels=embed_dim, bottleneck_dim=bottleneck, alpha=0.5)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, grid, cls_tok):
        adapted = self.adapter(grid)
        pooled = adapted.mean(dim=[-2, -1])
        return self.fc(cls_tok + pooled)


def load_system(device='cpu'):
    # Load DINOv2 backbone silently
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    class_names = DEFAULT_CLASSES
    if os.path.exists(CACHE_FILE):
        try:
            cache = torch.load(CACHE_FILE, map_location='cpu', weights_only=False)
            class_names = cache.get('class_names', DEFAULT_CLASSES)
        except Exception:
            pass

    num_classes = len(class_names)
    model = MorphicRoomClassifier(embed_dim=384, bottleneck=32, num_classes=num_classes).to(device)

    if os.path.exists(MODEL_FILE):
        state_dict = torch.load(MODEL_FILE, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
    else:
        print(f" [Warning] {MODEL_FILE} not found. Running in uninitialized mode.")

    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return backbone, model, class_names, transform


def predict_image(image_path: str, backbone, model, class_names, transform, device='cpu') -> bool:
    clean_path = image_path.strip().strip('"').strip("'")
    if not clean_path:
        return False

    if not os.path.exists(clean_path):
        print("\n" + "!" * 62)
        print("  [ERROR] File not found!")
        print(f"  Given path: {clean_path}")
        print("  Please verify the file path and try again.")
        print("!" * 62)
        return False

    try:
        img = Image.open(clean_path).convert("RGB")
    except Exception as e:
        print(f"\n  [ERROR] Unable to open image file: {e}")
        return False

    start_time = time.time()
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = backbone.forward_features(tensor)
        grid = feats['x_norm_patchtokens'].view(1, 16, 16, 384).permute(0, 3, 1, 2)
        cls_tok = feats['x_norm_clstoken']
        logits = model(grid, cls_tok)
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = torch.argmax(probs).item()

    inference_ms = (time.time() - start_time) * 1000.0
    pred_room = class_names[pred_idx]
    confidence = probs[pred_idx].item() * 100.0

    parts = pred_room.split('_')
    floor_display = parts[0] if len(parts) > 1 else "Building"
    room_display = parts[1] if len(parts) > 1 else pred_room

    print("\n" + "=" * 62)
    print(f"  AI DETECTION RESULT ({inference_ms:.1f} ms on CPU)")
    print("=" * 62)
    print(f"  BUILDING FLOOR  : {floor_display}")
    print(f"  DETECTED ROOM   : >> {room_display} <<")
    print(f"  CONFIDENCE      : {confidence:.2f}%")
    print("-" * 62)
    print("  Top 3 Probability Candidates:")
    top3_vals, top3_idx = torch.topk(probs, k=min(3, len(class_names)))
    for rank, (val, idx) in enumerate(zip(top3_vals, top3_idx), 1):
        c_name = class_names[idx.item()]
        c_parts = c_name.split('_')
        c_fl = c_parts[0] if len(c_parts) > 1 else ""
        c_rm = c_parts[1] if len(c_parts) > 1 else c_name
        bar_len = int(val.item() * 24)
        bar = "#" * bar_len
        print(f"    [{rank}] Room {c_rm:<5} ({c_fl:<9}) : {val.item()*100:5.1f}%  |{bar}")
    print("=" * 62)
    return True


def interactive_loop():
    print()
    print("=" * 62)
    print("      MORPHIC-CV: INTERACTIVE ROOM DETECTOR (CMD)")
    print("=" * 62)
    print(" [*] Loading AI model weights into RAM...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone, model, class_names, transform = load_system(device)

    print(" [*] Model ready! Fast CPU inference enabled.")
    print("=" * 62)
    print("  - Drag & drop an image or paste its path, then press Enter.")
    print("  - Type 'q' anytime to exit.")
    print("=" * 62)

    while True:
        print()
        try:
            user_input = input("Paste image file path (or 'q' to quit) > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting detector. Goodbye!")
            break

        if not user_input:
            continue

        clean_input = user_input.strip().strip('"').strip("'")
        if clean_input.lower() in ['q', 'quit', 'exit']:
            print("\nExiting detector. Goodbye!")
            break

        # Detect the room
        predict_image(clean_input, backbone, model, class_names, transform, device)

        # Ask if the user wants to upload/test another image
        print()
        try:
            choice = input("Would you like to test another image? [Y/n] (or 'q' to quit): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting detector. Goodbye!")
            break

        if choice in ['n', 'no', 'q', 'quit', 'exit']:
            print("\nExiting detector. Goodbye!")
            break
        # Any other key (or just pressing Enter) loops back to paste another image!


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Interactive Room Predictor")
    parser.add_argument("--image", type=str, default=None, help="Optional direct path to an image")
    args = parser.parse_args()

    if args.image:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        backbone, model, class_names, transform = load_system(device)
        predict_image(args.image, backbone, model, class_names, transform, device)
    else:
        interactive_loop()
