import os
from PIL import Image
from ultralytics import YOLO

HOME_DIR = os.path.expanduser("~")
SRC_DATASET_DIR = os.path.join(HOME_DIR, 'Downloads', 'birds_dataset')
DST_DATASET_DIR = os.path.join(HOME_DIR, 'Downloads', 'birds_dataset_openmv')

TARGET_SIZE = (320, 240)
BIRD_CLASS_ID = 14
print("[*] Loading YOLOv8 Object Detection Model...")
model = YOLO('yolov8n.pt')

def process_dataset():
    if not os.path.exists(SRC_DATASET_DIR):
        print(f"[X] Source directory not found at: {SRC_DATASET_DIR}")
        return

    print(f"[*] Pipeline initialized.")
    print(f"[*] Source: {SRC_DATASET_DIR}")
    print(f"[*] Destination: {DST_DATASET_DIR}")

    for class_name in os.listdir(SRC_DATASET_DIR):
        src_class_folder = os.path.join(SRC_DATASET_DIR, class_name)
        if not os.path.isdir(src_class_folder) or class_name.startswith('.'):
            continue
        dst_class_folder = os.path.join(DST_DATASET_DIR, class_name)
        os.makedirs(dst_class_folder, exist_ok=True)
        print(f"\n[+] Processing class: '{class_name}'")
        image_files = [f for f in os.listdir(src_class_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        success_count = 0
        fallback_count = 0

        for file_name in image_files:
            src_file_path = os.path.join(src_class_folder, file_name)
            dst_file_path = os.path.join(dst_class_folder, file_name)
            try:
                results = model(src_file_path, verbose=False)[0]
                boxes = results.boxes
                bird_boxes = [box for box in boxes if int(box.cls[0]) == BIRD_CLASS_ID]
                img = Image.open(src_file_path).convert('RGB')
                width, height = img.size

                if bird_boxes:
                    best_box = max(bird_boxes, key=lambda b: float(b.conf[0]))
                    xyxy = best_box.xyxy[0].tolist()
                    xmin, ymin, xmax, ymax = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    pad_w = int((xmax - xmin) * 0.4)
                    pad_h = int((ymax - ymin) * 0.4)
                    xmin = max(0, xmin - pad_w)
                    ymin = max(0, ymin - pad_h)
                    xmax = min(width, xmax + pad_w)
                    ymax = min(height, ymax + pad_h)
                    cropped_img = img.crop((xmin, ymin, xmax, ymax))
                    final_img = cropped_img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    final_img.save(dst_file_path, quality=85)
                    success_count += 1
                else:
                    min_dim = min(width, height)
                    xmin = (width - min_dim) // 2
                    ymin = (height - min_dim) // 2
                    xmax = xmin + min_dim
                    ymax = ymin + min_dim
                    cropped_img = img.crop((xmin, ymin, xmax, ymax))
                    final_img = cropped_img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    final_img.save(dst_file_path, quality=85)
                    fallback_count += 1

            except Exception as e:
                print(f" [X] Error processing {file_name}: {e}")

        print(f" [V] Finished '{class_name}': {success_count} smart crops, {fallback_count} center-crop fallbacks.")


if __name__ == "__main__":
    process_dataset()