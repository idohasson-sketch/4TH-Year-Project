import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Force usage of Apple Silicon GPU (MPS) or fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[*] Using runtime hardware device: {device}")

# --- PATHS & CONFIGURATION ---
HOME_DIR = os.path.expanduser("~")
DATASET_DIR = os.path.join(HOME_DIR, 'Downloads', 'birds_dataset')
MODEL_OUTPUT_PATH = os.path.join(HOME_DIR, 'Downloads', 'model.pt')
ONNX_OUTPUT_PATH = os.path.join(HOME_DIR, 'Downloads', 'model.onnx')

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 15

# --- DATA AUGMENTATION ---
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- LOADING DATASET ---
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"Database directory not found at: {DATASET_DIR}")

full_dataset = datasets.ImageFolder(root=DATASET_DIR, transform=transform_train)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"[V] Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
print(f"[V] Detected Classes: {full_dataset.classes}")

# --- TRANSFER LEARNING MODEL ---
print("[+] Initializing MobileNetV2...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(num_ftrs, 3)
)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# --- TRAINING LOOP ---
print("[+] Starting Training Loop...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_acc = correct / total
    print(f"Epoch {epoch + 1}/{EPOCHS} -> Loss: {running_loss/len(train_dataset):.4f} | Acc: {epoch_acc:.4f}")

# --- SAVE & EXPORT ---
torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
model.eval()
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)

torch.onnx.export(model, dummy_input, ONNX_OUTPUT_PATH, export_params=True,
                  opset_version=11, input_names=['input'], output_names=['output'])

print(f"[V] Model exported successfully to: {ONNX_OUTPUT_PATH}")
