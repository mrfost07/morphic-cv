"""
Production Template: Run Inference using a Trained MorphicCV Adapter.

How to use:
   python templates/inference_predict.py --image path/to/photo.jpg --adapter my_adapter.pt --classes cracked clean

Takes any new photo and outputs the predicted class and confidence percentage!
"""

import argparse
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from morphic_cv import MorphicDINOv2


def predict_single_image(image_path: str, adapter_path: str, class_names: list, model_name: str = "dinov2_vits14"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found!")
        return
    if not os.path.exists(adapter_path):
        print(f"Error: Adapter checkpoint {adapter_path} not found!")
        return

    print("=" * 65)
    print("MORPHIC-CV: PRODUCTION INFERENCE ENGINE")
    print(f"Image:   {image_path}")
    print(f"Adapter: {adapter_path} ({os.path.getsize(adapter_path)/1024:.1f} KB)")
    print("=" * 65)

    # 1. Transform input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    # 2. Load model & load lightweight adapter checkpoint
    num_classes = len(class_names)
    model = MorphicDINOv2.from_pretrained(model_name, num_classes=num_classes).to(device)
    model.load_adapter(adapter_path)
    model.eval()

    # 3. Predict
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = torch.argmax(probs).item()

    pred_class = class_names[pred_idx]
    confidence = probs[pred_idx].item() * 100.0

    print(f"\nPrediction Result: >> {pred_class.upper()} << ({confidence:.2f}% confidence)")
    print("-" * 65)
    print("Class Probabilities:")
    for i, name in enumerate(class_names):
        print(f"  [{i}] {name:<15}: {probs[i].item()*100:5.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MorphicCV Inference Script")
    parser.add_argument("--image", type=str, default="5_real_dinov2_benchmark/dog1.jpg", help="Path to input photo")
    parser.add_argument("--adapter", type=str, default="dinov2_morphic_adapter.pt", help="Path to adapter checkpoint")
    parser.add_argument("--classes", nargs="+", default=["class_0", "class_1", "class_2", "class_3"], help="Class names")
    parser.add_argument("--model", type=str, default="dinov2_vits14", help="DINOv2 model variant")
    args = parser.parse_args()

    predict_single_image(args.image, args.adapter, args.classes, model_name=args.model)
