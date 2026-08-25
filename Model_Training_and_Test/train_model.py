"""
===============================================================================
A-EYE TRACKER — Production Two-Stage Fine-Tuning MobileNetV2 Pipeline
===============================================================================
"""

import os
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from typing import List

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class TieredBirdDataset(Dataset):
    def __init__(self, base_dir: str, categories: List[str], tiers: List[str], transform=None):
        self.samples = []
        self.transform = transform
        self.categories = categories

        for class_idx, category in enumerate(categories):
            for tier in tiers:
                tier_folder = os.path.join(base_dir, category, f"{tier}_FT")
                if not os.path.exists(tier_folder):
                    tier_folder = os.path.join(base_dir, category, tier)

                if not os.path.exists(tier_folder):
                    continue

                for fname in os.listdir(tier_folder):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(tier_folder, fname), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def train_model(
    dataset_dir: str, 
    categories: List[str], 
    tiers: List[str], 
    output_weights: str, 
    epochs: int = 20, 
    batch_size: int = 16
):
    print("==================================================")
    print(f"[*] SOTA Two-Stage Fine-Tuning Pipeline on: {device}")
    print(f"[*] Base Directory: {dataset_dir}")
    print(f"[*] Target Classes ({len(categories)}): {categories}")
    print(f"[*] Training Tiers: {tiers}")
    print(f"[*] Target Output: {output_weights}")
    print("==================================================")

    # Balanced edge-tailored augmentations
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = TieredBirdDataset(
        base_dir=dataset_dir,
        categories=categories,
        tiers=tiers,
        transform=transform_train
    )

    if len(train_dataset) == 0:
        raise RuntimeError(f"[X] Error: No training images found in {dataset_dir} for tiers {tiers}.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    print(f"[+] Loaded {len(train_dataset)} training samples.")

    # Base Model Loading
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Custom Classifier Head
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(num_ftrs, len(categories))
    )

    # Phase 1: Freeze all backbone layers
    for param in model.features.parameters():
        param.requires_grad = False

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Optimizer with Warmup configuration
    optimizer = optim.AdamW([
        {'params': model.classifier.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4}
    ])
    
    warmup_epochs = min(5, epochs // 3)
    unfrozen_epochs = epochs - warmup_epochs

    best_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())

    print(f"\n--- Stage 1: Classifier Head Warmup ({warmup_epochs} Epochs) ---")
    for epoch in range(warmup_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100
        print(f"Warmup Epoch {epoch+1:02d}/{warmup_epochs:02d} -> Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

    print(f"\n--- Stage 2: Fine-Tuning Top Convolutional Blocks ({unfrozen_epochs} Epochs) ---")
    # Unfreeze top feature extraction layers (blocks 14-18)
    for param in model.features[14:].parameters():
        param.requires_grad = True

    # Differential Learning Rates
    optimizer = optim.AdamW([
        {'params': model.features[14:].parameters(), 'lr': 5e-5, 'weight_decay': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 3e-4, 'weight_decay': 1e-4}
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=unfrozen_epochs, eta_min=1e-6)

    for epoch in range(unfrozen_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_weights = copy.deepcopy(model.state_dict())

        print(f"Fine-Tune Epoch {epoch+1:02d}/{unfrozen_epochs:02d} -> Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% (Best Loss: {best_loss:.4f})")

    # Load and save the optimal state
    model.load_state_dict(best_model_weights)
    
    weights_dir = os.path.dirname(os.path.abspath(output_weights))
    if weights_dir:
        os.makedirs(weights_dir, exist_ok=True)
        
    torch.save(model.state_dict(), output_weights)
    print(f"\n[V] Best model state saved -> {output_weights}")

    # Export ONNX
    onnx_path = output_weights.replace('.pt', '.onnx')
    model.eval()
    dummy_input = torch.randn(1, 3, 128, 128, device=device)
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,
        opset_version=11, 
        input_names=['input'], 
        output_names=['output']
    )
    print(f"[V] Deployable ONNX model exported -> {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A-EYE Tracker MobileNetV2 SOTA Training Pipeline")
    parser.add_argument("--tiers", type=str, default="T1-1000", help="Comma-separated tiers (e.g. T1-1000,T2-150)")
    parser.add_argument("--categories", type=str, default="House_Sparrow,Feral_Pigeon,Rose_ringed_Parakeet,Hooded_Crow,Other")
    parser.add_argument("--output_weights", type=str, default="model_weights.pt")
    parser.add_argument("--dataset_dir", type=str, default=os.path.expanduser("~/Desktop/DB"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    train_model(
        dataset_dir=args.dataset_dir,
        categories=[c.strip() for c in args.categories.split(",") if c.strip()],
        tiers=[t.strip() for t in args.tiers.split(",") if t.strip()],
        output_weights=args.output_weights,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
