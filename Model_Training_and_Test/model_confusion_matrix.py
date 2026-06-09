import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
HOME_DIR = os.path.expanduser("~")
MODEL_PATH = os.path.join(HOME_DIR, 'Downloads', 'model.pt')
DB_DIR = "/Users/idohasson/Downloads/4TH-Year-Project/DB"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
classes = ['Feral_Pigeon', 'House_Sparrow', 'Other']

# Initialize model
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

all_targets = []
all_predictions = []

print(f"[*] Running full evaluation...")

for root, dirs, files in os.walk(DB_DIR):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Ground Truth
            name_lower = filename.lower()
            if "house_sparrow" in name_lower:
                expected = "House_Sparrow"
            elif "feral_pigeon" in name_lower:
                expected = "Feral_Pigeon"
            else:
                expected = "Other"

            img = Image.open(os.path.join(root, filename)).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_t)
                _, predicted_idx = torch.max(outputs, 1)

            all_targets.append(classes.index(expected))
            all_predictions.append(predicted_idx.item())

# Create Confusion Matrix
cm = confusion_matrix(all_targets, all_predictions)

# Plotting
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix: Model Performance')
plt.show()

# Print accuracy
accuracy = np.trace(cm) / np.sum(cm) * 100
print(f"Final Accuracy: {accuracy:.2f}%")
