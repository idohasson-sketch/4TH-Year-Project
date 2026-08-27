"""
===============================================================================
A-EYE TRACKER — Master End-to-End Orchestrator Pipeline
===============================================================================
Authors: Hodaya Hariri & Ido Hasson (Group 106)
Project: A-EYE Tracker: Smart Edge Wildlife Monitoring System
Platform: OpenMV Cam (STM32N6) & PyTorch / ONNX / TFLite Edge Stack

Description:
    Unified entry point executing all 4 consecutive phases:
    1. Phase 1: Dataset acquisition from iNaturalist API (T1 & T2 tiers)
    2. Phase 2: High-quality YOLOv8 object centering with 40% padding (*_center)
    3. Phase 3: Hardware downscaling to 128x128 @ 75% JPEG quality (*_FT)
    4. Phase 4: Two-Stage training, INT8 quantization, and Master Benchmark Suite (8x30 Matrix)
===============================================================================
"""

import os
import sys
import argparse
import subprocess
import time
from typing import List

# Exact script filenames as structured in the repository
SCRIPT_PHASE_1 = "phase_1:_dataset_collector_from_iNaturalist.py"
SCRIPT_PHASE_2 = "phase_2:_data_processing_centering_full_quality.py"
SCRIPT_PHASE_3 = "phase_3:_downscale_center_images.py"
SCRIPT_PHASE_4 = "evaluate_model.py"

def run_command(cmd: List[str], stage_name: str) -> None:
    """Executes a subprocess stage with runtime tracking and error handling."""
    print(f"\n{'='*75}")
    print(f"🚀 [STAGE: {stage_name}]")
    print(f"⚙️  Executing: {' '.join(cmd)}")
    print(f"{'='*75}")
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start_time
    if result.returncode != 0:
        print(f"\n[X] CRITICAL FAILURE in stage: '{stage_name}' (Exit Code: {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n[V] Completed '{stage_name}' successfully in {elapsed:.1f}s.")

def main():
    parser = argparse.ArgumentParser
    (
        description="A-EYE Tracker Master Pipeline Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Workspace directory paths
    parser.add_argument
   (
        "--dataset_dir",
        type=str,
        default=os.path.expanduser("~/Desktop/DB"),
        help="Root path for the multi-tiered dataset."
    )
    parser.add_argument
    (
        "--output_plots_dir",
        type=str,
        default=os.path.expanduser("~/Desktop/evaluation_plots"),
        help="Destination directory for benchmark plots and matrices."
    )
    parser.add_argument
    (
        "--models_dir",
        type=str,
        default=os.path.expanduser("~/Desktop/exported_models"),
        help="Directory to store ONNX and quantized INT8 models."
    )

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per fine-tuning stage.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for dataloaders.")

    # Execution control flags
    parser.add_argument("--skip_mining", action="store_true", help="Skip Phase 1 (iNaturalist download).")
    parser.add_argument("--skip_preprocessing", action="store_true", help="Skip Phase 2 & Phase 3 (Centering and Downscaling).")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip Phase 4 (Training and Benchmark evaluation).")
    args = parser.parse_args()

    # Create destination directories
    os.makedirs(args.dataset_dir, exist_ok=True)
    os.makedirs(args.output_plots_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    print("="*80)
    print("🌟 A-EYE TRACKER — MASTER PIPELINE INITIALIZATION")
    print(f"[*] Dataset Root: {args.dataset_dir}")
    print(f"[*] Exported Models:{args.models_dir}")
    print(f"[*] Output Figures: {args.output_plots_dir}")
    print(f"[*] Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print("="*80)

    # -------------------------------------------------------------------------
    # PHASE 1: Dataset Acquisition
    # -------------------------------------------------------------------------
    if not args.skip_mining:
        run_command([sys.executable, SCRIPT_PHASE_1], "Phase 1: iNaturalist Dataset Mining")
    else:
        print("\n[-] Skipping Phase 1: Mining (--skip_mining supplied).")

    # -------------------------------------------------------------------------
    # PHASE 2 & 3: Vision Preprocessing & Hardware Alignment
    # -------------------------------------------------------------------------
    if not args.skip_preprocessing:
        run_command([sys.executable, SCRIPT_PHASE_2], "Phase 2: YOLOv8 Centering & Cropping")
        run_command([sys.executable, SCRIPT_PHASE_3], "Phase 3: OpenMV 128x128 Downscaling & Compression")
    else:
        print("\n[-] Skipping Phases 2 & 3: Preprocessing (--skip_preprocessing supplied).")

    # -------------------------------------------------------------------------
    # PHASE 4: Two-Stage Training, Quantization & Master Evaluation
    # -------------------------------------------------------------------------
    if not args.skip_evaluation:
        run_command([sys.executable, SCRIPT_PHASE_4], "Phase 4: Two-Stage Fine-Tuning & Master Benchmark Matrix")
    else:
        print("\n[-] Skipping Phase 4: Training & Evaluation (--skip_evaluation supplied).")
    print("\n" + "="*80)
    print("🎉 MASTER PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"[V] Master Heatmap Matrix & Visual Comparisons -> {args.output_plots_dir}")
    print(f"[V] Quantized Edge INT8 Models & Checkpoints     -> {args.models_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
