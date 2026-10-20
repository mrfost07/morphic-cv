"""
Audit Script: Test & Verify the Accuracy of the Trained Model on Real Photos.
Performs:
1. Feature extraction & caching (saves 'building_features_cache.pt' for instant reuse).
2. Training MorphicCV Room Model and saving 'thesis_room_model.pt'.
3. Real individual image audit: takes actual test photos, prints predicted room vs true room,
   confidence %, and top-3 ranked alternatives.
4. Generates a visual verification image 'audit_predictions_visual.png' displaying
   the actual test photos with their prediction badges!
"""

import os
import time
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from morphic_cv.adapters.msa import MorphicAdapter


DATASET_DIR = r"C:\Users\fosta\OneDrive\Desktop\THESIS\Noemel\Noemel Dataset\New EB Building Images-20260905T175851Z-1-001\New EB Building Images"
CACHE_FILE = "building_features_cache.pt"
MODEL_FILE = "thesis_room_model.pt"


def discover_room_dataset(base_dir: str):
    room_images = defaultdict(list)
    for root, dirs, files in os.walk(base_dir):
        img_files = [os.path.join(root, f) for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if img_files:
            room_name = os.path.basename(root)
            floor_name = os.path.basename(os.path.dirname(root))
            full_class = f"{floor_name}_{room_name}"
            room_images[full_class].extend(img_files)

    class_names = sorted(list(room_images.keys()))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    return room_images, class_names, class_to_idx


def get_or_extract_features(room_images, class_to_idx, train_per_class=11, device='cpu'):
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached features from {CACHE_FILE}...")
        cache = torch.load(CACHE_FILE, map_location='cpu', weights_only=False)
        return cache['train_data'], cache['test_data'], cache['test_paths'], cache['class_names']

    print("Extracting features using official DINOv2 backbone...")
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_grids, train_cls, train_y = [], [], []
    test_grids, test_cls, test_y = [], [], []
    test_paths = []

    total_processed = 0
    with torch.no_grad():
        for class_name, img_paths in room_images.items():
            label = class_to_idx[class_name]
            sorted_paths = sorted(img_paths)
            train_paths = sorted_paths[:train_per_class]
            cur_test_paths = sorted_paths[train_per_class:]

            for path in train_paths:
                img = Image.open(path).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(device)
                feats = backbone.forward_features(tensor)
                grid = feats['x_norm_patchtokens'].view(1, 16, 16, 384).permute(0, 3, 1, 2).cpu()
                cls_tok = feats['x_norm_clstoken'].cpu()
                train_grids.append(grid)
                train_cls.append(cls_tok)
                train_y.append(label)
                total_processed += 1

            for path in cur_test_paths:
                img = Image.open(path).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(device)
                feats = backbone.forward_features(tensor)
                grid = feats['x_norm_patchtokens'].view(1, 16, 16, 384).permute(0, 3, 1, 2).cpu()
                cls_tok = feats['x_norm_clstoken'].cpu()
                test_grids.append(grid)
                test_cls.append(cls_tok)
                test_y.append(label)
                test_paths.append(path)
                total_processed += 1

            if (total_processed % 150) == 0 or total_processed == 690:
                print(f"  Processed {total_processed}/690 images...")

    train_data = {
        'grids': torch.cat(train_grids, dim=0),
        'cls': torch.cat(train_cls, dim=0),
        'labels': torch.tensor(train_y, dtype=torch.long)
    }
    test_data = {
        'grids': torch.cat(test_grids, dim=0),
        'cls': torch.cat(test_cls, dim=0),
        'labels': torch.tensor(test_y, dtype=torch.long)
    }

    class_names = sorted(list(class_to_idx.keys()))
    torch.save({
        'train_data': train_data,
        'test_data': test_data,
        'test_paths': test_paths,
        'class_names': class_names
    }, CACHE_FILE)
    print(f"Features cached to {CACHE_FILE} successfully!")
    return train_data, test_data, test_paths, class_names


class MorphicRoomClassifier(nn.Module):
    def __init__(self, embed_dim=384, bottleneck=32, num_classes=46):
        super().__init__()
        self.adapter = MorphicAdapter(in_channels=embed_dim, bottleneck_dim=bottleneck, alpha=0.5)
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, grid, cls_tok):
        adapted = self.adapter(grid)
        pooled = adapted.mean(dim=[-2, -1])
        return self.fc(cls_tok + pooled)


def train_room_model(train_data, num_classes, epochs=30, device='cpu'):
    print(f"\nTraining MorphicCV Room Model ({epochs} Epochs)...")
    model = MorphicRoomClassifier(embed_dim=384, bottleneck=32, num_classes=num_classes).to(device)
    train_ds = TensorDataset(train_data['grids'], train_data['cls'], train_data['labels'])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for ep in range(epochs):
        for b_grid, b_cls, b_y in train_loader:
            b_grid, b_cls, b_y = b_grid.to(device), b_cls.to(device), b_y.to(device)
            optimizer.zero_grad()
            out = model(b_grid, b_cls)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), MODEL_FILE)
    print(f"Model checkpoint saved to {MODEL_FILE} ({os.path.getsize(MODEL_FILE)/1024:.1f} KB)")
    return model


def audit_test_samples(model, test_data, test_paths, class_names, num_inspect=10, device='cpu'):
    print("\n" + "=" * 80)
    print("INDIVIDUAL TEST PHOTO AUDIT: VERIFYING PREDICTION ACCURACY")
    print("=" * 80)

    model.eval()
    grids = test_data['grids'].to(device)
    cls_toks = test_data['cls'].to(device)
    true_labels = test_data['labels'].to(device)

    with torch.no_grad():
        logits = model(grids, cls_toks)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

    # Calculate overall exact test accuracy
    correct_count = (preds == true_labels).sum().item()
    total_count = len(true_labels)
    acc = (correct_count / total_count) * 100.0
    print(f"Overall Verified Test Accuracy: {acc:.2f}% ({correct_count} / {total_count} images)\n")

    # Pick samples spread evenly across the test set (covering 2nd, 3rd, 4th, 5th floors)
    step = len(test_paths) // num_inspect
    sample_indices = [i * step for i in range(num_inspect)]

    audit_cards = []
    print(f"{'#':<3} | {'True Room':<18} | {'Predicted Room':<18} | {'Confidence':<10} | {'Status':<8} | {'Top-3 Alternatives'}")
    print("-" * 80)

    for i, idx in enumerate(sample_indices):
        true_name = class_names[true_labels[idx].item()]
        pred_name = class_names[preds[idx].item()]
        conf = probs[idx, preds[idx].item()].item() * 100.0
        is_correct = (true_name == pred_name)
        status = "MATCH" if is_correct else "MISMATCH"

        # Top 3
        top3_vals, top3_idx = torch.topk(probs[idx], k=3)
        top3_str = ", ".join([f"{class_names[c.item()]} ({top3_vals[j].item()*100:.1f}%)" for j, c in enumerate(top3_idx)])

        print(f"{i+1:<3} | {true_name:<18} | {pred_name:<18} | {conf:6.2f}%    | {status:<8} | {top3_str}")
        audit_cards.append({
            'path': test_paths[idx],
            'true': true_name,
            'pred': pred_name,
            'conf': conf,
            'correct': is_correct
        })

    print("=" * 80)

    # Generate Visual Proof Image
    create_visual_proof_image(audit_cards, "audit_predictions_visual.png")


def create_visual_proof_image(cards, output_path):
    """Stitches test photos into a visual diagnostic verification collage."""
    num_cards = len(cards)
    cols = 5
    rows = (num_cards + cols - 1) // cols
    tile_w, tile_h = 240, 200

    composite = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(25, 25, 25))
    draw = ImageDraw.Draw(composite)

    for i, card in enumerate(cards):
        r = i // cols
        c = i % cols
        x = c * tile_w
        y = r * tile_h

        # Load and resize real photo
        try:
            img = Image.open(card['path']).convert("RGB").resize((tile_w - 10, tile_h - 55))
            composite.paste(img, (x + 5, y + 5))
        except Exception:
            pass

        # Draw status banner
        banner_color = (30, 180, 50) if card['correct'] else (220, 50, 50)
        draw.rectangle([x + 5, y + tile_h - 48, x + tile_w - 5, y + tile_h - 5], fill=banner_color)

        draw.text((x + 8, y + tile_h - 45), f"TRUE: {card['true']}", fill="white")
        draw.text((x + 8, y + tile_h - 26), f"PRED: {card['pred']} ({card['conf']:.1f}%)", fill="white")

    composite.save(output_path)
    print(f"\nVisual proof collage saved to: {output_path}")
    print("Open 'audit_predictions_visual.png' to see the test photos with their prediction badges!")


if __name__ == '__main__':
    room_images, class_names, class_to_idx = discover_room_dataset(DATASET_DIR)
    train_data, test_data, test_paths, class_names = get_or_extract_features(room_images, class_to_idx, train_per_class=11)
    model = train_room_model(train_data, num_classes=len(class_names), epochs=30)
    audit_test_samples(model, test_data, test_paths, class_names, num_inspect=10)
