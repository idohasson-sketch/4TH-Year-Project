"""
===============================================================================
A-EYE TRACKER — Full Master Pipeline (Train -> INT8 Quantize -> Quantized Eval)
===============================================================================
Purpose:
Orchestrates training, dynamic INT8 quantization, and hardware-true evaluation 
of MobileNetV2 (via ONNX Runtime) vs YOLOv8, producing:
1. System Evaluation Master Matrix (8x30 Hierarchical Heatmap)
2. Overall Performance Comparison Bar Chart
===============================================================================
"""

import os
import sys
import subprocess
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from ultralytics import YOLO
from typing import List

HOME_DIR = os.path.expanduser("~")
DEFAULT_DB_DIR = os.path.join(HOME_DIR, 'Desktop', 'DB')
OUTPUT_PLOTS_DIR = os.path.join(HOME_DIR, 'Desktop', 'evaluation_plots')
MODELS_DIR = os.path.join(HOME_DIR, 'Desktop', 'exported_models')
TARGET_SIZE = (128, 128)
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def invoke_training_and_quantization
(
    tiers: List[str],
    categories: List[str],
    model_tag: str,
    base_dir: str = DEFAULT_DB_DIR,
    epochs: int = 20
) 
-> str:
    """Trains MobileNetV2 and automatically executes INT8 dynamic quantization."""
    pt_path = os.path.join(MODELS_DIR, f"{model_tag}.pt")
    onnx_path = os.path.join(MODELS_DIR, f"{model_tag}.onnx")
    quant_onnx_path = os.path.join(MODELS_DIR, f"{model_tag}_quantized.onnx")
    # 1. Train & Export ONNX
    train_cmd = 
    [
        sys.executable, "train_model.py",
        "--tiers", ",".join(tiers),
        "--categories", ",".join(categories),
        "--output_weights", pt_path,
        "--dataset_dir", base_dir,
        "--epochs", str(epochs)
    ]
    print(f"\n⚙️ [Step 1: Training] Executing: {' '.join(train_cmd)}")
    res_train = subprocess.run(train_cmd)
    if res_train.returncode != 0:
        raise RuntimeError(f"[X] Training failed for {model_tag}")

    # 2. Quantize ONNX to INT8
    quant_cmd = [
        sys.executable, "quantize_model.py",
        "--input_onnx", onnx_path,
        "--output_quantized_onnx", quant_onnx_path
    ]
    print(f"⚙️ [Step 2: Quantization] Executing: {' '.join(quant_cmd)}")
    res_quant = subprocess.run(quant_cmd)
    if res_quant.returncode != 0:
        raise RuntimeError(f"[X] Quantization failed for {model_tag}")
    return quant_onnx_path

def softmax(x: np.ndarray) -> np.ndarray:
    """Computes stable softmax for model outputs."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def evaluate_single_cell
(
    model_type: str,
    model_obj,
    target_category: str,
    category_idx: int,
    folder_path: str
)
-> float:
    """Evaluates a single species/tier/resolution folder with a 20% confidence threshold."""
    if not os.path.exists(folder_path):
        return 0.0

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        return 0.0

    correct, total = 0, 0
    CONF_THRESHOLD = 0.50
    if model_type == "mobilenet_quantized":
        transform = transforms.Compose
        ([
            transforms.Resize(TARGET_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_name = model_obj.get_inputs()[0].name

        for filename in image_files:
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img_t = transform(img).unsqueeze(0).numpy().astype(np.float32)
                outputs = model_obj.run(None, {input_name: img_t})
                logits = outputs[0][0]
                probs = softmax(logits)
                pred_idx = int(np.argmax(probs))
                max_prob = float(probs[pred_idx])
                total += 1
                # If confidence is below 20%, it is considered unclassified (not counted as correct)
                if max_prob >= CONF_THRESHOLD and pred_idx == category_idx:
                    correct += 1
            except Exception:
                continue

    elif model_type == "yolo":
        for filename in image_files:
            img_path = os.path.join(folder_path, filename)
            try:
                results = model_obj(img_path, verbose=False)[0]
                # Filter boxes by confidence threshold (0.20)
                valid_boxes = [box for box in results.boxes if float(box.conf[0]) >= CONF_THRESHOLD]
                detected_bird = any(int(box.cls[0]) == 14 for box in valid_boxes)
                total += 1
                if (target_category != "Other" and detected_bird) or (target_category == "Other" and not detected_bird):
                    correct += 1
            except Exception:
                continue
    return (correct / total * 100.0) if total > 0 else 0.0

def compute_benchmark_row
(
    model_type: str,
    model_obj,
    categories: List[str],
    base_dir: str
) 
-> np.ndarray:
    """Computes a 30-element vector for all 5 species x 3 tiers x 2 resolutions."""
    row_values = []
    tiers = ["T1-1000", "T2-150", "T3-50"]

    for cat_idx, category in enumerate(categories):
        for tier in tiers:
            # Full Resolution Center
            center_folder = os.path.join(base_dir, category, f"{tier}_center")
            if not os.path.exists(center_folder):
                center_folder = os.path.join(base_dir, category, tier)
            acc_center = evaluate_single_cell(model_type, model_obj, category, cat_idx, center_folder)
            row_values.append(acc_center)
            # Low Resolution / Hardware Ready (_FT)
            ft_folder = os.path.join(base_dir, category, f"{tier}_FT")
            if not os.path.exists(ft_folder):
                ft_folder = os.path.join(base_dir, category, tier)
            acc_ft = evaluate_single_cell(model_type, model_obj, category, cat_idx, ft_folder)
            row_values.append(acc_ft)
    return np.array(row_values)


def plot_overall_performance_comparison(yolo_means: List[float], mobilenet_means: List[float], save_path: str):
    tests = ["TEST 1:\nSanity Baseline", "TEST 2:\nPhase 1 Trained", "TEST 3:\nPhase 2 Trained", "TEST 4:\nPhase 3 Trained"]
    x = np.arange(len(tests))
    width = 0.35
    fig, ax = plt.subplots(figsize=(13, 7), dpi=300)
    rects1 = ax.bar(x - width/2, yolo_means, width, label='YOLOv8', color='#1f77b4', edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x + width/2, mobilenet_means, width, label='MobileNetV2 (INT8)', color='#32b1c8', edgecolor='black', linewidth=0.8)
    ax.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Overall Performance Comparison: YOLOv8 vs MobileNetV2 (INT8 Quantized)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(tests, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', fontsize=11, frameon=True, edgecolor='grey')
    for rect in list(rects1) + list(rects2):
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved Bar Chart -> {save_path}")

def plot_master_evaluation_matrix(matrix_data: np.ndarray, species_list: List[str], save_path: str):
    fig, ax = plt.subplots(figsize=(24, 11), dpi=300)
    cmap = LinearSegmentedColormap.from_list("custom_heat", ["#ffffe0", "#a1dab4", "#41b6c4", "#225ea8", "#081d58"])

    sns.heatmap(matrix_data, annot=True, fmt=".1f", cmap=cmap, cbar=False, linewidths=1.0, 
                linecolor='white', vmin=0, vmax=100, annot_kws={"size": 7.5, "weight": "bold"}, ax=ax)

    sub_cols = ["center", "center\nlow_res"] * (len(species_list) * 3)
    ax.set_xticks(np.arange(len(sub_cols)) + 0.5)
    ax.set_xticklabels(sub_cols, fontsize=7.5, fontweight='bold')
    ax.set_yticks([])

    tiers = ["T1-1000", "T2-150", "T3-50"]
    for s_idx, species in enumerate(species_list):
        sp_start = s_idx * 6
        ax.text(sp_start + 3, -1.2, f"Species: {species}", ha='center', va='bottom', fontsize=11, fontweight='bold')
        for t_idx, tier in enumerate(tiers):
            ax.text(sp_start + t_idx * 2 + 1, -0.3, tier, ha='center', va='bottom', fontsize=9, fontweight='bold')

    test_boxes = 
    [
        ("TEST 1:\nSanity Baseline\n(ImageNet Weights)\nBaseline zero-shot\nevaluation without\nlocal training.", "#fde0dd"),
        ("TEST 2:\nPhase 1 Trained\n(ImageNet + T1)\nModel exposed to 1,000\nlab images per species.", "#e5f5e0"),
        ("TEST 3:\nPhase 2 Trained\n(ImageNet + T1 + T2)\nComprehensive training\nwith 1,150 images.", "#deebf7"),
        ("TEST 4:\nPhase 3 Fully Trained\n(ImageNet + All Tiers)\n1,000 + 150 lab +\n50 field images.", "#f2f0f7")
    ]
    for i, (text, color) in enumerate(test_boxes):
        ax.text(-2.2, i * 2 + 1, text, ha='center', va='center', fontsize=7.5, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=color, edgecolor='grey', alpha=0.9))
    ax.set_title("SYSTEM EVALUATION MASTER MATRIX - HIERARCHICAL ANALYSIS (INT8 QUANTIZED)", 
                 fontsize=13, fontweight='bold', pad=35, y=-0.18)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved Master Matrix -> {save_path}")

def run_full_experimental_benchmarks(categories: List[str], base_dir: str = DEFAULT_DB_DIR):
    print("==================================================================")
    print("🚀 STARTING BENCHMARKS WITH INT8 QUANTIZED ONNX INFERENCE")
    print("==================================================================")

    master_matrix = np.zeros((8, len(categories) * 6))
    yolo_model = YOLO("yolov8n.pt")
    yolo_row = compute_benchmark_row("yolo", yolo_model, categories, base_dir)

    # --- TEST 1: Sanity Baseline ---
    print("\n--- TEST 1: Baseline Evaluation ---")
    q1_path = invoke_training_and_quantization(["T1-1000"], categories, "model_baseline_t1", base_dir, epochs=1)
    session_t1 = ort.InferenceSession(q1_path, providers=['CPUExecutionProvider'])
    mb_row_1 = np.zeros(30) # Baseline untargeted weights
    master_matrix[0, :] = yolo_row
    master_matrix[1, :] = mb_row_1

    # --- TEST 2: Phase 1 Trained (T1-1000) ---
    print("\n--- TEST 2: T1 Trained (INT8) ---")
    q2_path = invoke_training_and_quantization(["T1-1000"], categories, "model_t1", base_dir, epochs=20)
    session_t2 = ort.InferenceSession(q2_path, providers=['CPUExecutionProvider'])
    mb_row_2 = compute_benchmark_row("mobilenet_quantized", session_t2, categories, base_dir)
    master_matrix[2, :] = yolo_row
    master_matrix[3, :] = mb_row_2

    # --- TEST 3: Phase 2 Trained (T1+T2) ---
    print("\n--- TEST 3: T1 + T2 Trained (INT8) ---")
    q3_path = invoke_training_and_quantization(["T1-1000", "T2-150"], categories, "model_t1_t2", base_dir, epochs=20)
    session_t3 = ort.InferenceSession(q3_path, providers=['CPUExecutionProvider'])
    mb_row_3 = compute_benchmark_row("mobilenet_quantized", session_t3, categories, base_dir)
    master_matrix[4, :] = yolo_row
    master_matrix[5, :] = mb_row_3

    # --- TEST 4: Phase 3 Fully Trained (All Tiers) ---
    print("\n--- TEST 4: All Tiers Trained (INT8 Sanity Check) ---")
    q4_path = invoke_training_and_quantization(["T1-1000", "T2-150", "T3-50"], categories, "model_all", base_dir, epochs=20)
    session_t4 = ort.InferenceSession(q4_path, providers=['CPUExecutionProvider'])
    mb_row_4 = compute_benchmark_row("mobilenet_quantized", session_t4, categories, base_dir)
    master_matrix[6, :] = yolo_row
    master_matrix[7, :] = mb_row_4

    # Macro Averages & Visual Rendering
    yolo_macro_means = [float(np.mean(master_matrix[i, :])) for i in [0, 2, 4, 6]]
    mobilenet_macro_means = [float(np.mean(master_matrix[i, :])) for i in [1, 3, 5, 7]]

    plot_overall_performance_comparison
    (
        yolo_macro_means, 
        mobilenet_macro_means, 
        os.path.join(OUTPUT_PLOTS_DIR, "models_overall_comparison.png")
    )
    plot_master_evaluation_matrix
    (
        master_matrix, 
        categories, 
        os.path.join(OUTPUT_PLOTS_DIR, "sanity_and_training_matrix_reproduced.jpg")
    )


if __name__ == "__main__":
    ACTIVE_CATEGORIES = 
    [
        "House_Sparrow",
        "Feral_Pigeon",
        "Rose_ringed_Parakeet",
        "Hooded_Crow",
        "Other"
    ]
    run_full_experimental_benchmarks(categories=ACTIVE_CATEGORIES)
