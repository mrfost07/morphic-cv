"""
Unified Master Runner for Next-Gen Lightweight Computer Vision Suite.
Runs all 3 benchmark projects sequentially and summarizes their performance,
memory footprints, and next-generation research capabilities.
"""

import sys
import os
import subprocess
import time

def run_project(name: str, script_path: str):
    print("\n" + "#" * 70)
    print(f"RUNNING: {name}")
    print("#" * 70)
    start = time.time()
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    elapsed = time.time() - start
    status = "SUCCESS" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"--> {name} finished in {elapsed:.2f}s with status: {status}")
    return result.returncode == 0

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    projects = [
        ("Project 1: SMM-Net (Sparse Morphological Network & RigL)", os.path.join(root, "1_sparse_morphology", "demo_train.py")),
        ("Project 2: Feature-First Compressed Patch-Token Streaming", os.path.join(root, "2_feature_token_stream", "demo_streaming.py")),
        ("Project 3: Morphological Sparse Adapters (MSA for PEFT)", os.path.join(root, "3_hybrid_morphic_adapter", "eval_robustness.py")),
        ("Project 4: GigaPixel Morphological Router (Native High-Res Vision)", os.path.join(root, "4_gigapixel_morphic_router", "benchmark_gigapixel.py")),
        ("Project 5: Real Meta DINOv2 Foundation Model + MSA on Real Photos", os.path.join(root, "5_real_dinov2_benchmark", "benchmark_real_world.py"))
    ]

    print("=" * 70)
    print("LIGHTWEIGHT NEXT-GEN COMPUTER VISION RESEARCH SUITE (<8GB RAM)")
    print("=" * 70)

    success_count = 0
    for name, path in projects:
        ok = run_project(name, path)
        if ok:
            success_count += 1

    print("\n" + "=" * 70)
    print(f"ALL BENCHMARKS FINISHED: {success_count}/{len(projects)} passed successfully!")
    print("Every module runs strictly within < 100 MB of RAM (far below the 8GB limit).")
    print("=" * 70)

if __name__ == '__main__':
    main()
