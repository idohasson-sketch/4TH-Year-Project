"""
===============================================================================
A-EYE TRACKER — Automated Training & Multi-Model Evaluation Pipeline
===============================================================================
Purpose:
Orchestrates the four core experimental benchmarks outlined in the research.
Prior to each supervised evaluation, it automatically executes 'train_model.py'
with the designated data tiers, dynamic categories, and output model weights,
subsequently evaluating both MobileNetV2 and YOLOv8 on hardware-aligned (_FT) sets.

Experimental Benchmarks:
1. Zero-Shot Baseline: Evaluates pre-trained/untrained models directly on T2-150.
2. T1 -> T2 Evaluation: Trains on T1-1000 via train_model.py, tests on T2-150.
3. T1+T2 -> Field Test: Trains on T1+T2 (1150 images), tests on field data (T3-50).
4. Full-Train Sanity Check: Trains on all data (1200 images, including T3),
   and evaluates on T3 to measure training stability/memorization.
===============================================================================
"""

import os
import sys
import subprocess
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from ultralytics import YOLO
from typing import List, Dict

# --- Path & Hardware Configurations ---
HOME_DIR = os.path.expanduser("~")
DEFAULT_DB_DIR = os.path.join(HOME_DIR, 'Desktop', 'DB')
TARGET_SIZE = (128, 128)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def invoke_training_script(
    tiers_to_train: List[str],
    categories: List[str],
    output_weight_path: str,
    base_dir: str = DEFAULT_DB_DIR,
    epochs: int = 20
) -> bool:
    """
    Executes 'train_model.py' as a subprocess with explicit command-line arguments.
    """
    script_path = "train_model.py"
    if not os.path.exists(script_path):
        print(f"[X] Training script '{script_path}' not found in current directory.")
        return False

    cmd = [
        sys.executable, script_path,
        "--tiers", ",".join(tiers_to_train),
        "--categories", ",".join(categories),
        "--output_weights", output_weight_path,
        "--dataset_dir", base_dir,
        "--epochs", str(epochs)
    ]

    print(f"\n⚙️ [Triggering Training Pipeline] Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"✅ Training completed successfully -> Weights saved to: {output_weight_path}")
        return True
    else:
        print(f"❌ Training failed with exit code: {result.returncode}")
        return False


def get_mobilenet_model(num_classes: int, model_weight_path: str = None) -> nn.Module:
    """Initializes MobileNetV2 architecture with a custom classification head."""
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    if model_weight_path and os.path.exists(model_weight_path):
        model.load_state_dict(torch.load(model_weight_path, map_location=DEVICE))
        print(f"[*] Loaded MobileNetV2 weights from: {model_weight_path}")
    model.to(DEVICE)
    model.eval()
    return model


def evaluate_mobilenet(
    model: nn.Module, 
    test_tier: str, 
    categories: List[str],
    base_dir: str = DEFAULT_DB_DIR
) -> Dict[str, float]:
    """
    Evaluates MobileNetV2 on hardware-aligned (_FT) directories for a specific tier.
    """
    transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    all_targets = []
    all_predictions = []

    print(f"[*] Evaluating MobileNetV2 on tier: {test_tier}")

    for idx, category in enumerate(categories):
        # Path format: DB/<Category>/<Tier>_FT
        cat_folder = os.path.join(base_dir, category, f"{test_tier}_FT")
        if not os.path.exists(cat_folder):
            # Fallback if _FT suffix is omitted or for field images
            cat_folder = os.path.join(base_dir, category, test_tier)

        if not os.path.exists(cat_folder):
            print(f"   [!] Folder not found: {cat_folder}")
            continue

        for filename in os.listdir(cat_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(cat_folder, filename)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_t = transform(img).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        outputs = model(img_t)
                        _, predicted_idx = torch.max(outputs, 1)

                    all_targets.append(idx)
                    all_predictions.append(predicted_idx.item())
                except Exception as e:
                    print(f"   [X] Error processing {filename}: {e}")

    if not all_targets:
        print("   [!] No images evaluated.")
        return {}

    cm = confusion_matrix(all_targets, all_predictions, labels=list(range(len(categories))))
    accuracy = np.trace(cm) / np.sum(cm) * 100
    
    print(f"   [V] Evaluation Finished. Accuracy: {accuracy:.2f}%")
    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "report": classification_report(all_targets, all_predictions, target_names=categories, zero_division=0)
    }


def evaluate_yolov8_baseline(
    yolo_model_path: str, 
    test_tier: str, 
    categories: List[str],
    base_dir: str = DEFAULT_DB_DIR
) -> Dict[str, float]:
    """
    Evaluates YOLOv8 detection model across the defined categories.
    """
    print(f"[*] Evaluating YOLOv8 Model on tier: {test_tier}")
    yolo_model = YOLO(yolo_model_path)
    
    all_targets = []
    all_predictions = []

    for idx, category in enumerate(categories):
        cat_folder = os.path.join(base_dir, category, f"{test_tier}_FT")
        if not os.path.exists(cat_folder):
            cat_folder = os.path.join(base_dir, category, test_tier)

        if not os.path.exists(cat_folder):
            continue

        for filename in os.listdir(cat_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(cat_folder, filename)
                try:
                    results = yolo_model(img_path, verbose=False)[0]
                    detected_bird = any(int(box.cls[0]) == 14 for box in results.boxes)
                    
                    pred_idx = idx if detected_bird else (0 if idx != 0 else 1)
                    all_targets.append(idx)
                    all_predictions.append(pred_idx)
                except Exception as e:
                    print(f"   [X] Error in YOLOv8 inference for {filename}: {e}")

    if not all_targets:
        return {}

    cm = confusion_matrix(all_targets, all_predictions, labels=list(range(len(categories))))
    accuracy = np.trace(cm) / np.sum(cm) * 100
    print(f"   [V] YOLOv8 Evaluation Complete. Accuracy: {accuracy:.2f}%")
    return {"accuracy": accuracy, "confusion_matrix": cm}


def run_experimental_benchmarks(
    categories: List[str], 
    base_dir: str = DEFAULT_DB_DIR
):
    """
    Executes the 4 experimental benchmarks, invoking train_model.py before benchmarks 2, 3, and 4.
    """
    print("==================================================================")
    print("🚀 STARTING A-EYE TRACKER TRAINING & BENCHMARK PIPELINE")
    print(f"Active Categories ({len(categories)}): {categories}")
    print("==================================================================")

    num_classes = len(categories)

    # --- BENCHMARK 1: Zero-Shot / No Training ---
    print("\n--- [Benchmark 1] Zero-Shot Evaluation (No Training) ---")
    zs_model = get_mobilenet_model(num_classes)
    evaluate_mobilenet(zs_model, test_tier="T2-150", categories=categories, base_dir=base_dir)
    evaluate_yolov8_baseline("yolov8n.pt", test_tier="T2-150", categories=categories, base_dir=base_dir)

    # --- BENCHMARK 2: Train on T1-1000 -> Test on T2-150 ---
    print("\n--- [Benchmark 2] Train on T1-1000 -> Test on T2-150 ---")
    weights_b2 = "model_weights_t1.pt"
    if invoke_training_script(tiers_to_train=["T1-1000"], categories=categories, output_weight_path=weights_b2, base_dir=base_dir):
        t1_model = get_mobilenet_model(num_classes, weights_b2)
        evaluate_mobilenet(t1_model, test_tier="T2-150", categories=categories, base_dir=base_dir)

    # --- BENCHMARK 3: Train on T1+T2 (1150) -> Test on Field T3-50 ---
    print("\n--- [Benchmark 3] Train on T1+T2 (1150) -> Test on Field T3-50 ---")
    weights_b3 = "model_weights_t1_t2.pt"
    if invoke_training_script(tiers_to_train=["T1-1000", "T2-150"], categories=categories, output_weight_path=weights_b3, base_dir=base_dir):
        t12_model = get_mobilenet_model(num_classes, weights_b3)
        evaluate_mobilenet(t12_model, test_tier="T3-50", categories=categories, base_dir=base_dir)

    # --- BENCHMARK 4: Train on All (1200) -> Test on T3-50 (Sanity Check) ---
    print("\n--- [Benchmark 4] Train on All Data (1200) -> Test on T3-50 (Sanity Check) ---")
    weights_b4 = "model_weights_all.pt"
    if invoke_training_script(tiers_to_train=["T1-1000", "T2-150", "T3-50"], categories=categories, output_weight_path=weights_b4, base_dir=base_dir):
        all_model = get_mobilenet_model(num_classes, weights_b4)
        evaluate_mobilenet(all_model, test_tier="T3-50", categories=categories, base_dir=base_dir)

    print("\n==================================================================")
    print("📊 ALL 4 EXPERIMENTAL BENCHMARKS COMPLETED")
    print("==================================================================")


if __name__ == "__main__":
    ACTIVE_CATEGORIES = [
        "House_Sparrow",
        "Feral_Pigeon",
        "Rose_ringed_Parakeet",
        "Hooded_Crow",
        "Other"
    ]
    
    run_experimental_benchmarks(categories=ACTIVE_CATEGORIES)
