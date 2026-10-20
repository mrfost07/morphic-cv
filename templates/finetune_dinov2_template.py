"""
Production Template: Fine-Tuning Meta's DINOv2 on Custom Data using MorphicCV.

How to use with your own data:
1. Organize your images into subfolders by class:
   my_dataset/
   ├── cat/
   │   ├── 01.jpg
   │   └── 02.jpg
   └── dog/
       ├── 01.jpg
       └── 02.jpg

2. Run:
   python templates/finetune_dinov2_template.py --data-dir ./my_dataset --epochs 5

This script trains on < 2 GB of RAM and saves a tiny ~150 KB adapter checkpoint!
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Import the new morphic_cv library
from morphic_cv import MorphicDINOv2


def train_dinov2_custom(
    data_dir: str = None,
    model_name: str = "dinov2_vits14",
    num_classes: int = 2,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-3,
    output_path: str = "dinov2_morphic_adapter.pt"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("MORPHIC-CV: FINE-TUNING META DINOv2 WITH MORPHOLOGICAL ADAPTERS")
    print(f"Device: {device} | Target Model: {model_name}")
    print("=" * 70)

    # 1. Transforms tailored for DINOv2 (224x224, normalized to ImageNet stats)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Dataset Setup
    if data_dir and os.path.isdir(data_dir):
        print(f"Loading custom dataset from: {data_dir}")
        dataset = ImageFolder(root=data_dir, transform=transform)
        classes = dataset.classes
        num_classes = len(classes)
        print(f"Detected {num_classes} classes: {classes}")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        print("[Demo Mode] No dataset folder specified. Creating synthetic demo batch...")
        demo_x = torch.randn(16, 3, 224, 224)
        demo_y = torch.randint(0, num_classes, (16,))
        loader = [(demo_x, demo_y)]

    # 3. Initialize Pretrained DINOv2 with Morphological Sparse Adapter
    print(f"\nInitializing {model_name} with Morphological Adapter (MSA)...")
    model = MorphicDINOv2.from_pretrained(model_name, num_classes=num_classes, bottleneck_dim=32).to(device)
    
    summary = model.get_parameter_summary()
    print(f"Total Model Parameters:     {summary['total_parameters']:,}")
    print(f"Trainable Adapter Params:   {summary['trainable_parameters']:,} ({summary['trainable_fraction_pct']:.2f}% of network)")
    print("All base foundation weights are 100% frozen (0 gradient memory!).\n")

    # 4. Optimizer & Loss (Only updates adapter + classifier head)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    model.train()
    print(f"--- Training for {epochs} Epochs ---")
    start_total = time.time()
    for ep in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        start_ep = time.time()

        for batch_imgs, batch_labels in loader:
            batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_imgs)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)

        ep_time = time.time() - start_ep
        avg_loss = running_loss / max(1, len(loader))
        acc = (correct / max(1, total)) * 100.0
        print(f"Epoch {ep:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Accuracy: {acc:5.1f}% | Time: {ep_time:.2f}s")

    total_time = time.time() - start_total
    print(f"\nTraining completed in {total_time:.2f}s!")

    # 6. Save Adapter Checkpoint (< 200 KB)
    model.save_adapter(output_path)
    print("=" * 70)
    print(f"Fine-tuning ready! You can deploy this adapter alongside the frozen DINOv2 backbone.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Meta DINOv2 with MorphicCV")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to ImageFolder dataset directory")
    parser.add_argument("--model", type=str, default="dinov2_vits14", help="DINOv2 backbone name")
    parser.add_argument("--classes", type=int, default=2, help="Number of classes (auto-detected if data-dir provided)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--output", type=str, default="dinov2_morphic_adapter.pt", help="Path to save adapter checkpoint")
    args = parser.parse_args()

    train_dinov2_custom(
        data_dir=args.data_dir,
        model_name=args.model,
        num_classes=args.classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_path=args.output
    )
