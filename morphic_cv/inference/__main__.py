"""
MorphicCV Universal Inference CLI.
Run on pure CPU with < 300MB RAM.

Usage examples:
    # 1. Interactive terminal loop
    python -m morphic_cv.inference --adapter model.pt --classes cat dog

    # 2. Single image classification
    python -m morphic_cv.inference --mode image --path photo.jpg --adapter model.pt --classes cat dog

    # 3. Batch folder of images
    python -m morphic_cv.inference --mode folder --path ./images/ --adapter model.pt --classes cat dog

    # 4. Face Detection & Recognition
    python -m morphic_cv.inference --mode faces --path group_photo.jpg --adapter model.pt --classes Alice Bob

    # 5. Salient Object Detection
    python -m morphic_cv.inference --mode salient --path photo.jpg

    # 6. Live Webcam Real-Time Stream (zero GPU)
    python -m morphic_cv.inference --mode webcam --camera 0
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from morphic_cv.inference.engine import MorphicInferenceEngine


def main():
    parser = argparse.ArgumentParser(description="MorphicCV Universal Inference CLI")
    parser.add_argument("--mode", type=str, default="interactive", choices=["interactive", "image", "folder", "faces", "salient", "webcam"], help="Inference mode")
    parser.add_argument("--path", type=str, default=None, help="Path to image file or directory")
    parser.add_argument("--adapter", type=str, default=None, help="Path to trained .pt adapter checkpoint")
    parser.add_argument("--classes", nargs="+", default=None, help="List of class names (e.g. --classes cat dog bird)")
    parser.add_argument("--model", type=str, default="dinov2_vits14", help="Base model backbone")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID for webcam streaming")
    parser.add_argument("--webcam-mode", type=str, default="classify", choices=["classify", "faces", "salient"], help="Webcam streaming detection mode")
    args = parser.parse_args()

    engine = MorphicInferenceEngine(
        model_name=args.model,
        adapter_path=args.adapter,
        class_names=args.classes,
    )

    if args.mode == "interactive":
        engine.interactive_cli()

    elif args.mode == "image":
        if not args.path:
            print("Error: --path required for image mode.")
            sys.exit(1)
        res = engine.predict(args.path, return_probs=True)
        print("\n" + "=" * 60)
        print(f"  PREDICTION: >> {res['label']} << ({res['confidence']:.2f}%) [{res['latency_ms']:.1f}ms]")
        print("=" * 60)

    elif args.mode == "folder":
        if not args.path or not os.path.isdir(args.path):
            print(f"Error: Directory not found at {args.path}")
            sys.exit(1)
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        files = [os.path.join(args.path, f) for f in os.listdir(args.path) if f.lower().endswith(exts)]
        print(f"Processing {len(files)} images from {args.path}...")
        results = engine.predict_batch(files)
        for fpath, r in zip(files, results):
            fname = os.path.basename(fpath)
            print(f"  {fname:<30} -> {r['label']:<15} ({r['confidence']:.1f}%) [{r['latency_ms']:.1f}ms]")

    elif args.mode == "faces":
        if not args.path:
            print("Error: --path required for faces mode.")
            sys.exit(1)
        faces = engine.detect_faces(args.path)
        print(f"\nDetected {len(faces)} face(s) in {args.path}:")
        for i, f in enumerate(faces, 1):
            print(f"  Face #{i}: BBox {f['bbox']} -> Label: {f['label']} ({f['confidence']:.1f}%)")

    elif args.mode == "salient":
        if not args.path:
            print("Error: --path required for salient mode.")
            sys.exit(1)
        objects = engine.detect_salient_objects(args.path)
        print(f"\nDetected {len(objects)} salient object(s) in {args.path}:")
        for i, obj in enumerate(objects, 1):
            print(f"  Object #{i}: BBox {obj['bbox']} -> Label: {obj['label']} ({obj['confidence']:.1f}%)")

    elif args.mode == "webcam":
        engine.stream_webcam(camera_id=args.camera, mode=args.webcam_mode)


if __name__ == "__main__":
    main()
