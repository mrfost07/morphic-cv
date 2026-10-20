"""
morphic_cv.cli.main
Command-line interface for MorphicCV toolkit.
"""

import argparse
import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
from ..models.dinov2 import MorphicDINOv2
from ..routing.router import SaliencyRouter
from .. import __version__


def handle_info(args):
    print(f"MorphicCV version: {__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print("MorphicCV is ready for ultra-low memory computer vision (<8GB RAM)!")


def handle_route(args):
    print(f"Routing high-resolution image: {args.image}")
    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} not found!")
        sys.exit(1)

    img = Image.open(args.image).convert('RGB')
    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0) # [1, 3, H, W]

    router = SaliencyRouter(patch_size=args.patch_size, keep_ratio=args.keep_ratio)
    selected_patches, coords, meta = router(tensor)

    print(f"Total Patches: {meta['total_patches']}")
    print(f"Kept Patches: {meta['kept_patches']} ({meta['sparsity_ratio']:.1f}% background pruned)")
    print(f"Attention Memory Reduction: {meta['attention_memory_reduction']:.1f}x")
    print("Saliency routing complete.")


def handle_finetune(args):
    print(f"Initializing MorphicDINOv2 from backbone: {args.model}")
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model = MorphicDINOv2.from_pretrained(args.model, num_classes=args.classes, bottleneck_dim=args.bottleneck)
    model.to(device)

    summary = model.get_parameter_summary()
    print(f"Total Parameters: {summary['total_parameters']:,}")
    print(f"Trainable Parameters: {summary['trainable_parameters']:,} ({summary['trainable_fraction_pct']:.2f}% of network)")
    print(f"Ready for fine-tuning on device: {device}")


def main():
    parser = argparse.ArgumentParser(prog="morphic-cv", description="MorphicCV: Next-Gen Lightweight Vision Toolkit (<8GB RAM)")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # info
    p_info = subparsers.add_parser("info", help="Display environment and version info")
    p_info.set_defaults(func=handle_info)

    # route
    p_route = subparsers.add_parser("route", help="Scan high-resolution image and extract salient patches")
    p_route.add_argument("--image", required=True, type=str, help="Path to high-resolution image")
    p_route.add_argument("--patch-size", default=16, type=int, help="Patch size (default 16)")
    p_route.add_argument("--keep-ratio", default=0.10, type=float, help="Ratio of salient patches to keep (default 0.10)")
    p_route.set_defaults(func=handle_route)

    # finetune
    p_tune = subparsers.add_parser("finetune", help="Initialize or fine-tune DINOv2 with Morphological Adapters")
    p_tune.add_argument("--model", default="dinov2_vits14", type=str, help="DINOv2 model variant")
    p_tune.add_argument("--classes", default=10, type=int, help="Number of classes")
    p_tune.add_argument("--bottleneck", default=32, type=int, help="Bottleneck dimension for MSA adapter")
    p_tune.add_argument("--cpu", action="store_true", help="Force CPU execution")
    p_tune.set_defaults(func=handle_finetune)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
