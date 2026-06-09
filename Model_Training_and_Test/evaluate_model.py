import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# path to model location
HOME_DIR = os.path.expanduser("~")
MODEL_PATH = os.path.join(HOME_DIR, 'Downloads', 'model.pt')
# path to tested files
DB_DIR = "/Users/idohasson/Downloads/4TH-Year-Project/birds_dataset_openmv"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Classes must match training order: Pigeon, Sparrow, Other
classes = ['Feral_Pigeon', 'House_Sparrow', 'Other']

# Model Architecture
model = models.mobilenet_v2()
model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.classifier[1].in_features, 3))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

correct = 0
total = 0

print(f"[*] Starting evaluation on {DB_DIR}...")

for root, dirs, files in os.walk(DB_DIR):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(root, filename)

            # Extract expected class from filename
            name_lower = filename.lower()
            if "house_sparrow" in name_lower:
                expected = "House_Sparrow"
            elif "feral_pigeon" in name_lower:
                expected = "Feral_Pigeon"
            else:
                expected = "Other"

            # Inference
            try:
                img = Image.open(full_path).convert('RGB')
                img_t = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(img_t)
                    _, predicted_idx = torch.max(outputs, 1)

                predicted_label = classes[predicted_idx.item()]

                total += 1
                is_correct = (predicted_label == expected)
                if is_correct:
                    correct += 1

                print(
                    f"File: {filename[:20]}... | Expected: {expected} | Got: {predicted_label} | {'✅' if is_correct else '❌'}")

            except Exception as e:
                print(f"[!] Error processing {filename}: {e}")

if total > 0:
    print(f"\nEvaluation Finished. Total: {total} | Accuracy: {(correct / total) * 100:.2f}%")
else:
    print("[!] No images found to evaluate.")
