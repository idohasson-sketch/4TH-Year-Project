"""
===============================================================================
A-EYE TRACKER — Smart Object-Centric Cropping & Dataset Standardization Pipeline
===============================================================================
Purpose:
Executes the second phase of data preparation by isolating and centering avian
subjects within raw downloaded imagery using a pre-trained YOLOv8 object detector.

Key Workflow & Logic:
1. Object Detection via Tiny/Nano Vision: Runs YOLOv8n inference on raw images to
   detect objects belonging to COCO class 14 ('bird').
2. Bounding-Box Padding: Identifies the highest-confidence bird bounding box and
   applies a dynamic 40% spatial padding on all sides to preserve anatomical context.
3. Fallback Centering: If no bird is detected with sufficient confidence, executes a
   standardized geometric center-crop to maintain consistent aspect ratios.
4. Lossless Quality Export: Exports cropped outputs at original camera resolution
   with maximum fidelity (JPEG quality=100) into dedicated '*_center' directories.
5. Strict Tier Filtering: Confined exclusively to Tier 1 ('T1-1000') and Tier 2 ('T2-150')
   per species, ignoring manual test splits and auxiliary folders.
===============================================================================
"""

import os
from PIL import Image
from ultralytics import YOLO

# --- Directory & Pipeline Configurations ---
HOME_DIR = os.path.expanduser("~")
SRC_DATASET_DIR = os.path.join(HOME_DIR, 'Desktop', 'DB')
# COCO Dataset Class Index for 'bird'
BIRD_CLASS_ID = 14
# Explicit tiers processed in this phase
VALID_TIERS = {"T1-1000", "T2-150"}

print("[*] Loading YOLOv8 Object Detection Model for High-Quality Centering...")
model = YOLO('yolov8n.pt')


def process_dataset_center(dataset_dir=SRC_DATASET_DIR):
    """
    Scans the dataset directory, applies YOLOv8-based smart cropping to valid tiers,
    and writes centered crops at full resolution into corresponding '*_center' folders.
    """
    if not os.path.exists(dataset_dir):
        print(f"[X] Source directory not found at: {dataset_dir}")
        return

    print(f"[*] High-Quality Centering Pipeline Initialized.")
    print(f"[*] Source: {dataset_dir}")
    print(f"[*] Target Tiers: {list(VALID_TIERS)}")
    print(f"[*] Mode: Original Resolution & Maximum Quality (quality=100)\n")

    # Iterate through each species / class directory
    for class_name in os.listdir(dataset_dir):
        src_class_folder = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(src_class_folder) or class_name.startswith('.'):
            continue

        print(f"\n==================================================")
        print(f"[+] Scanning Tiers for class: '{class_name}'")
        print(f"==================================================")

        for tier_name in os.listdir(src_class_folder):
            src_tier_folder = os.path.join(src_class_folder, tier_name)
            # Ensure processing only the 2 specified tiers (T1-1000 and T2-150)
            if not os.path.isdir(src_tier_folder) or tier_name not in VALID_TIERS:
                continue
            dst_tier_folder = os.path.join(src_class_folder, f"{tier_name}_center")
            os.makedirs(dst_tier_folder, exist_ok=True)
            print(f" [->] Processing '{tier_name}' -> Creating '{tier_name}_center'...")

            image_files = [f for f in os.listdir(src_tier_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
            success_count = 0
            fallback_count = 0

            for file_name in image_files:
                src_file_path = os.path.join(src_tier_folder, file_name)
                dst_file_path = os.path.join(dst_tier_folder, file_name)
                if os.path.exists(dst_file_path):
                    success_count += 1
                    continue

                try:
                    # Run YOLOv8 detection
                    results = model(src_file_path, verbose=False)[0]
                    boxes = results.boxes
                    bird_boxes = [box for box in boxes if int(box.cls[0]) == BIRD_CLASS_ID]
                    img = Image.open(src_file_path).convert('RGB')
                    width, height = img.size
                    if bird_boxes:
                        # Select the bird bounding box with highest confidence
                        best_box = max(bird_boxes, key=lambda b: float(b.conf[0]))
                        xyxy = best_box.xyxy[0].tolist()
                        xmin, ymin, xmax, ymax = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                        # Add 40% proportional padding
                        pad_w = int((xmax - xmin) * 0.4)
                        pad_h = int((ymax - ymin) * 0.4)
                        xmin = max(0, xmin - pad_w)
                        ymin = max(0, ymin - pad_h)
                        xmax = min(width, xmax + pad_w)
                        ymax = min(height, ymax + pad_h)
                        cropped_img = img.crop((xmin, ymin, xmax, ymax))
                        cropped_img.save(dst_file_path, 'JPEG', quality=100)
                        success_count += 1
                    else:
                        # Fallback: square crop from center
                        min_dim = min(width, height)
                        xmin = (width - min_dim) // 2
                        ymin = (height - min_dim) // 2
                        xmax = xmin + min_dim
                        ymax = ymin + min_dim
                        cropped_img = img.crop((xmin, ymin, xmax, ymax))
                        cropped_img.save(dst_file_path, 'JPEG', quality=100)
                        fallback_count += 1

                except Exception as e:
                    print(f"   [X] Error processing {file_name}: {e}")
            print(f" [V] Completed {tier_name}_center: {success_count} smart crops, {fallback_count} center fallbacks.")


if __name__ == "__main__":
    process_dataset_center()
