"""
Production Template: Native High-Resolution (2K/4K) Saliency Routing on < 2GB RAM.

How to use:
   python templates/gigapixel_inference.py --image path/to/large_photo.jpg --keep-ratio 0.05

Extracts the top 5% most salient geometric patches in milliseconds without OOM!
"""

import argparse
import os
import time
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms
from morphic_cv import SaliencyRouter


def process_gigapixel(image_path: str, patch_size: int = 16, keep_ratio: float = 0.05, output_vis: str = "routed_output.png"):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} does not exist!")
        return

    print("=" * 70)
    print("MORPHIC-CV: NATIVE GIGAPIXEL SALIENCY ROUTER")
    print(f"Loading image: {image_path}")
    print("=" * 70)

    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size

    # Crop/pad so dimensions are multiples of patch_size
    new_w = (orig_w // patch_size) * patch_size
    new_h = (orig_h // patch_size) * patch_size
    if new_w != orig_w or new_h != orig_h:
        img = img.crop((0, 0, new_w, new_h))

    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0) # [1, 3, H, W]
    print(f"Processing Resolution: {new_w} x {new_h} pixels")

    router = SaliencyRouter(patch_size=patch_size, keep_ratio=keep_ratio)

    start_time = time.time()
    selected_patches, coords, meta = router(tensor)
    elapsed_ms = (time.time() - start_time) * 1000.0

    print(f"\nSaliency Scan Elapsed Time: {elapsed_ms:.2f} ms")
    print(f"Total Patches:             {meta['total_patches']:,}")
    print(f"Active Kept Patches:       {meta['kept_patches']:,} ({meta['sparsity_ratio']:.1f}% background pruned!)")
    print(f"Attention Memory Reduction:{meta['attention_memory_reduction']:.1f}x less attention footprint!")

    # Draw bounding boxes on the kept patches to visualize what the router selected
    vis_img = img.copy()
    draw = ImageDraw.Draw(vis_img)
    topk = meta['topk_indices'][0].tolist()
    grid_w = new_w // patch_size

    for idx in topk:
        py = (idx // grid_w) * patch_size
        px = (idx % grid_w) * patch_size
        draw.rectangle([px, py, px + patch_size, py + patch_size], outline="cyan", width=1)

    vis_img.save(output_vis)
    print(f"\nVisualization saved to: {output_vis}")
    print("Cyan boxes indicate the high-entropy structural patches selected for sparse processing.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GigaPixel Image Saliency Router")
    parser.add_argument("--image", type=str, default="5_real_dinov2_benchmark/astronaut.jpg", help="Path to input image")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size")
    parser.add_argument("--keep-ratio", type=float, default=0.10, help="Fraction of patches to keep")
    parser.add_argument("--output", type=str, default="routed_output.png", help="Output path for visualization")
    args = parser.parse_args()

    process_gigapixel(args.image, patch_size=args.patch_size, keep_ratio=args.keep_ratio, output_vis=args.output)
