import os
from ultralytics import YOLO
from supabase import create_client, Client

SUPABASE_URL = "https://pxkevqlcaiazhgqrxbsp.supabase.co"
SUPABASE_KEY = "sb_publishable_igo1VwAo9FEGetZssdbFZQ_gOBRLIi3"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
model_baseline = YOLO("yolov8n.pt")
BIRD_CLASS_ID = 14
BASE_DB_DIR = "/Users/idohasson/Downloads/4TH-Year-Project/birds_dataset_openmv"
SUB_FOLDERS = ["House Sparrow", "Feral Pigeon"]


def run_baseline_inference(image_path):
    try:
        results = model_baseline(image_path, verbose=False)
        detected_classes = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id == BIRD_CLASS_ID:
                    detected_classes.append("bird")
                else:
                    detected_classes.append(model_baseline.names[class_id])
        return detected_classes
    except Exception as e:
        print(f"Error running inference on {image_path}: {e}")
        return None

for folder_name in SUB_FOLDERS:
    folder_path = os.path.join(BASE_DB_DIR, folder_name)
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue
    print(f"\n--- Processing folder: {folder_name} ---")
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(folder_path, filename)
            predictions = run_baseline_inference(full_path)
            if predictions is None:
                continue
            is_correct = "bird" in predictions
            data = \
                {
                "image_name": filename,
                "expected_species": folder_name,
                "lens_top_predictions": predictions,
                "is_correct": is_correct
            }

            try:
                response = supabase.table("yolov8_comparison_table_OpenMVN6_quality").insert(data).execute()
                print(f"Uploaded: {filename} | Detected: {predictions} | Correct: {is_correct}")
            except Exception as e:
                print(f"Failed to upload {filename}: {e}")
