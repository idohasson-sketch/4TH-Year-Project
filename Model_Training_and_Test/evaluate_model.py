"""
===============================================================================
A-EYE TRACKER — Master Automated Benchmark & Visual Evaluation Suite
===============================================================================
Purpose:
Orchestrates the 4 core experimental evaluation benchmarks, calling 'train_model.py'
before each supervised training stage. It dynamically computes real inference
accuracy metrics across all species, data tiers (T1, T2, T3), and resolutions 
(full-resolution '_center' vs downscaled '_FT' hardware-ready sets).

Visual Artifacts Generated:
1. System Evaluation Master Matrix (Hierarchical Heatmap: 8 rows x 30 columns)
2. Overall Performance Comparison (Grouped Bar Chart: YOLOv8 vs MobileNetV2)
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
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from ultralytics import YOLO
from typing import List, Dict, Tuple

# --- Path & Runtime Configurations ---
HOME_DIR = os.path.expanduser("~")
DEFAULT_DB_DIR = os.path.join(HOME_DIR, 'Desktop', 'DB')
OUTPUT_PLOTS_DIR = os.path.join(HOME_DIR, 'Desktop', 'evaluation_plots')
TARGET_SIZE = (128, 128)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)


def invoke_training_script(
    tiers_to_train: List[str],
    categories: List[str],
    output_weight_path: str,
    base_dir: str = DEFAULT_DB_DIR,
    epochs: int = 20
) -> bool:
    """Executes 'train_model.py' as a subprocess with strict arguments."""
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
    return result.returncode == 0


def get_mobilenet_model(num_classes: int, model_weight_path: str = None) -> nn.Module:
    """Initializes MobileNetV2 with custom classification head."""
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


def evaluate_single_cell(
    model_type: str,
    model_obj,
    target_category: str,
    category_idx: int,
    folder_path: str
) -> float:
    """
    Computes true accuracy (%) for a single cell (specific species, tier, and resolution).
    """
    if not os.path.exists(folder_path):
        return 0.0

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        return 0.0

    correct = 0
    total = 0

    if model_type == "mobilenet":
        transform = transforms.Compose([
            transforms.Resize(TARGET_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        for filename in image_files:
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img_t = transform(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    outputs = model_obj(img_t)
                    _, pred_idx = torch.max(outputs, 1)
                total += 1
                if pred_idx.item() == category_idx:
                    correct += 1
            except Exception:
                continue

    elif model_type == "yolo":
        for filename in image_files:
            img_path = os.path.join(folder_path, filename)
            try:
                results = model_obj(img_path, verbose=False)[0]
                detected_bird = any(int(box.cls[0]) == 14 for box in results.boxes)
                total += 1
                if (target_category != "Other" and detected_bird) or (target_category == "Other" and not detected_bird):
                    correct += 1
            except Exception:
                continue

    return (correct / total * 100.0) if total > 0 else 0.0


def compute_benchmark_row(
    model_type: str,
    model_obj,
    categories: List[str],
    base_dir: str
) -> np.ndarray:
    """
    Computes a 30-element vector representing performance across all species,
    tiers (T1-1000, T2-150, T3-50), and variants (center vs center_low_res / _FT).
    """
    row_values = []
    tiers = ["T1-1000", "T2-150", "T3-50"]

    for cat_idx, category in enumerate(categories):
        for tier in tiers:
            # 1. Full Resolution Centered Crop
            center_folder = os.path.join(base_dir, category, f"{tier}_center")
            if not os.path.exists(center_folder):
                center_folder = os.path.join(base_dir, category, tier)
            acc_center = evaluate_single_cell(model_type, model_obj, category, cat_idx, center_folder)
            row_values.append(acc_center)

            # 2. Downscaled Hardware Profile (_FT)
            ft_folder = os.path.join(base_dir, category, f"{tier}_FT")
            if not os.path.exists(ft_folder):
                ft_folder = os.path.join(base_dir, category, tier)
            acc_ft = evaluate_single_cell(model_type, model_obj, category, cat_idx, ft_folder)
            row_values.append(acc_ft)

    return np.array(row_values)


def plot_overall_performance_comparison(
    yolo_means: List[float], 
    mobilenet_means: List[float], 
    save_path: str
):
    """Renders the macro performance comparison grouped bar chart."""
    tests = [
        "TEST 1:\nSanity Baseline",
        "TEST 2:\nPhase 1 Trained",
        "TEST 3:\nPhase 2 Trained",
        "TEST 4:\nPhase 3 Trained"
    ]
    
    x = np.arange(len(tests))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 7), dpi=300)
    rects1 = ax.bar(x - width/2, yolo_means, width, label='YOLOv8', color='#1f77b4', edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x + width/2, mobilenet_means, width, label='MobileNetV2', color='#32b1c8', edgecolor='black', linewidth=0.8)

    ax.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Overall Performance Comparison: YOLOv8 vs MobileNetV2', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(tests, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', fontsize=11, frameon=True, edgecolor='grey')

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved Figure 1 (Bar Chart) -> {save_path}")


def plot_master_evaluation_matrix(
    matrix_data: np.ndarray, 
    species_list: List[str], 
    save_path: str
):
    """Renders the complete 8x30 Hierarchical Heatmap with exact thesis styling."""
    fig, ax = plt.subplots(figsize=(24, 11), dpi=300)
    
    cmap = LinearSegmentedColormap.from_list("custom_heat", ["#ffffe0", "#a1dab4", "#41b6c4", "#225ea8", "#081d58"])

    sns.heatmap(
        matrix_data, 
        annot=True, 
        fmt=".1f", 
        cmap=cmap, 
        cbar=False, 
        linewidths=1.0, 
        linecolor='white',
        vmin=0, 
        vmax=100, 
        annot_kws={"size": 7.5, "weight": "bold"},
        ax=ax
    )

    sub_cols = ["center", "center\nlow_res"] * (len(species_list) * 3)
    ax.set_xticks(np.arange(len(sub_cols)) + 0.5)
    ax.set_xticklabels(sub_cols, fontsize=7.5, fontweight='bold')
    ax.set_yticks([])

    tiers = ["T1-1000", "T2-150", "T3-50"]
    for s_idx, species in enumerate(species_list):
        sp_start = s_idx * 6
        sp_mid = sp_start + 3
        ax.text(sp_mid, -1.2, f"Species: {species}", ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        for t_idx, tier in enumerate(tiers):
            t_mid = sp_start + t_idx * 2 + 1
            ax.text(t_mid, -0.3, tier, ha='center', va='bottom', fontsize=9, fontweight='bold')

    test_boxes = [
        ("TEST 1:\nSanity Baseline\n(ImageNet Weights)\nBaseline zero-shot\nevaluation without\nlocal training.", "#fde0dd"),
        ("TEST 2:\nPhase 1 Trained\n(ImageNet + T1)\nModel exposed to 1,000\nlab images per species.", "#e5f5e0"),
        ("TEST 3:\nPhase 2 Trained\n(ImageNet + T1 + T2)\nComprehensive training\nwith 1,150 images.", "#deebf7"),
        ("TEST 4:\nPhase 3 Fully Trained\n(ImageNet + All Tiers)\n1,000 + 150 lab +\n50 field images.", "#f2f0f7")
    ]

    for i, (text, color) in enumerate(test_boxes):
        y_pos = i * 2 + 1
        ax.text(-2.2, y_pos, text, ha='center', va='center', fontsize=7.5, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=color, edgecolor='grey', alpha=0.9))

    ax.set_title("SYSTEM EVALUATION MASTER MATRIX - HIERARCHICAL ANALYSIS", 
                 fontsize=13, fontweight='bold', pad=35, y=-0.18)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved Figure 2 (Master Matrix) -> {save_path}")


def run_full_experimental_benchmarks(
    categories: List[str],
    base_dir: str = DEFAULT_DB_DIR
):
    """
    Executes the entire automated benchmark and generates true metric visual figures.
    """
    print("==================================================================")
    print("🚀 INITIALIZING COMPLETE A-EYE TRACKER EXPERIMENTAL PIPELINE")
    print(f"Categories ({len(categories)}): {categories}")
    print(f"Target DB: {base_dir}")
    print("==================================================================")

    num_classes = len(categories)
    # Master matrix: 8 rows (4 tests x 2 models) by 30 columns (5 classes x 3 tiers x 2 variants)
    master_matrix = np.zeros((8, len(categories) * 6))

    yolo_model = YOLO("yolov8n.pt")
    yolo_row_1 = compute_benchmark_row("yolo", yolo_model, categories, base_dir)

    # --- TEST 1: Sanity Baseline (Zero-Shot) ---
    print("\n--- Running TEST 1: Sanity Baseline (Zero-Shot) ---")
    zs_mobilenet = get_mobilenet_model(num_classes)
    mb_row_1 = compute_benchmark_row("mobilenet", zs_mobilenet, categories, base_dir)
    master_matrix[0, :] = yolo_row_1
    master_matrix[1, :] = mb_row_1

    # --- TEST 2: Phase 1 Trained (T1-1000) ---
    print("\n--- Running TEST 2: Phase 1 Trained (T1-1000) ---")
    weights_p1 = "model_weights_t1.pt"
    if invoke_training_script(["T1-1000"], categories, weights_p1, base_dir):
        mb_p1 = get_mobilenet_model(num_classes, weights_p1)
        mb_row_2 = compute_benchmark_row("mobilenet", mb_p1, categories, base_dir)
    else:
        mb_row_2 = np.zeros(30)
    master_matrix[2, :] = yolo_row_1
    master_matrix[3, :] = mb_row_2

    # --- TEST 3: Phase 2 Trained (T1-1000 + T2-150) ---
    print("\n--- Running TEST 3: Phase 2 Trained (T1 + T2) ---")
    weights_p2 = "model_weights_t1_t2.pt"
    if invoke_training_script(["T1-1000", "T2-150"], categories, weights_p2, base_dir):
        mb_p2 = get_mobilenet_model(num_classes, weights_p2)
        mb_row_3 = compute_benchmark_row("mobilenet", mb_p2, categories, base_dir)
    else:
        mb_row_3 = np.zeros(30)
    master_matrix[4, :] = yolo_row_1
    master_matrix[5, :] = mb_row_3

    # --- TEST 4: Phase 3 Fully Trained (All Tiers) ---
    print("\n--- Running TEST 4: Phase 3 Fully Trained (All Tiers) ---")
    weights_p3 = "model_weights_all.pt"
    if invoke_training_script(["T1-1000", "T2-150", "T3-50"], categories, weights_p3, base_dir):
        mb_p3 = get_mobilenet_model(num_classes, weights_p3)
        mb_row_4 = compute_benchmark_row("mobilenet", mb_p3, categories, base_dir)
    else:
        mb_row_4 = np.zeros(30)
    master_matrix[6, :] = yolo_row_1
    master_matrix[7, :] = mb_row_4

    # Extract averages for the overall bar chart
    yolo_macro_means = [float(np.mean(master_matrix[i, :])) for i in [0, 2, 4, 6]]
    mobilenet_macro_means = [float(np.mean(master_matrix[i, :])) for i in [1, 3, 5, 7]]

    # Render Visual Figures
    plot_overall_performance_comparison(
        yolo_macro_means, 
        mobilenet_macro_means, 
        os.path.join(OUTPUT_PLOTS_DIR, "models_overall_comparison.png")
    )

    plot_master_evaluation_matrix(
        master_matrix, 
        categories, 
        os.path.join(OUTPUT_PLOTS_DIR, "sanity_and_training_matrix_reproduced.jpg")
    )

    print("\n==================================================================")
    print("🎯 BENCHMARK SUITE AND PLOT GENERATION FINISHED SUCCESSFULLY")
    print(f"All outputs saved to: {OUTPUT_PLOTS_DIR}")
    print("==================================================================")


if __name__ == "__main__":
    ACTIVE_CATEGORIES = [
        "House_Sparrow",
        "Feral_Pigeon",
        "Rose_ringed_Parakeet",
        "Hooded_Crow",
        "Other"
    ]
    run_full_experimental_benchmarks(categories=ACTIVE_CATEGORIES)
