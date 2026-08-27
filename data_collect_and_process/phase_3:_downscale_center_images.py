"""
===============================================================================
A-EYE TRACKER — Phase 3: Hardware-Aligned Downscaling & Compression Pipeline
===============================================================================
Purpose:
Takes the centered, full-quality images produced in Phase 2 (*_center folders)
and applies spatial downscaling and JPEG compression to match the exact hardware
input constraints of the OpenMV N6 TinyML model.

Key Workflow & Logic:
1. Input Ingestion: Scans Phase 2 output directories ('T1-1000_center' and
   'T2-150_center') across all classes.
2. High-Quality Spatial Resampling: Downsamples each image to 128x128 pixels using
   PIL's Lanczos anti-aliasing filter (Image.Resampling.LANCZOS).
3. Memory Footprint Optimization: Encodes images at 75% JPEG quality to emulate
   edge camera buffer constraints.
4. Output Export: Stores processed datasets into dedicated '*_FT' folders ready
   for training and INT8 quantization.
===============================================================================
"""
import os
from PIL import Image

# --- Directory & Hardware Target Configurations ---
HOME_DIR = os.path.expanduser("~")
SRC_DATASET_DIR = os.path.join(HOME_DIR, 'Desktop', 'DB')
TARGET_SIZE = (128, 128)
JPEG_QUALITY = 75

# Input tiers from Phase 2 to be downscaled
SOURCE_TIERS = {"T1-1000_center", "T2-150_center"}

def downscale_dataset(dataset_dir=SRC_DATASET_DIR):
    """
    Iterates through Phase 2 centered folders, resizes all images to 128x128,
    and saves them to corresponding '*_FT' directories with 75% quality.
    """
    if not os.path.exists(dataset_dir):
        print(f"[X] Source directory not found at: {dataset_dir}")
        return

    print(f"[*] Hardware Downscaling Pipeline Initialized.")
    print(f"[*] Source Root: {dataset_dir}")
    print(f"[*] Target Profile: OpenMV N6 ({TARGET_SIZE[0]}x{TARGET_SIZE[1]}, JPEG Quality: {JPEG_QUALITY}%)\n")

    # Iterate through each species / class folder
    for class_name in os.listdir(dataset_dir):
        src_class_folder = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(src_class_folder) or class_name.startswith('.'):
            continue

        print(f"\n==================================================")
        print(f"[+] Scanning Phase 2 Tiers for class: '{class_name}'")
        print(f"==================================================")
        for folder_name in os.listdir(src_class_folder):
            src_tier_folder = os.path.join(src_class_folder, folder_name)
            # Process only centered folders from Phase 2
            if not os.path.isdir(src_tier_folder) or folder_name not in SOURCE_TIERS:
                continue

            # Map e.g. 'T1-1000_center' -> 'T1-1000_FT'
            base_tier_name = folder_name.replace('_center', '')
            dst_tier_folder = os.path.join(src_class_folder, f"{base_tier_name}_FT")
            os.makedirs(dst_tier_folder, exist_ok=True)
            print(f" [->] Downscaling '{folder_name}' -> '{base_tier_name}_FT'...")
            image_files = [f for f in os.listdir(src_tier_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
            processed_count = 0
            for file_name in image_files:
                src_file_path = os.path.join(src_tier_folder, file_name)
                dst_file_path = os.path.join(dst_tier_folder, file_name)
                if os.path.exists(dst_file_path):
                    processed_count += 1
                    continue
                try:
                    with Image.open(src_file_path) as img:
                        img_rgb = img.convert('RGB')
                        downscaled_img = img_rgb.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                        downscaled_img.save(dst_file_path, 'JPEG', quality=JPEG_QUALITY)
                        processed_count += 1
                except Exception as e:
                    print(f"   [X] Error processing {file_name}: {e}")
            print(f" [V] Completed {base_tier_name}_FT: {processed_count}/{len(image_files)} images downscaled.")


if __name__ == "__main__":
    downscale_dataset()
