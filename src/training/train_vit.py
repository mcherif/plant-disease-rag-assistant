"""
train_classifier.py

This script trains a Vision Transformer (ViT) model for plant disease classification
using the Hugging Face Transformers and PyTorch libraries. The model is trained on a
custom image dataset and logs metrics and artifacts using MLflow.

Key Features:
-------------
- Loads and preprocesses images from a local `ImageFolder` dataset.
- Uses a pretrained ViT (`google/vit-base-patch16-224`) from Hugging Face Transformers.
- Performs training and validation with configurable hyperparameters.
- Logs training and validation metrics (accuracy, F1-score, loss) to an MLflow tracking server.
- Saves the fine-tuned model and processor to `models/vit-finetuned/`.

Configuration:
--------------
- Dataset must be stored under `data/split/` with `train/` and `val/` subfolders.
- Trained model is saved to `models/vit-finetuned/`.
- Requires MLflow server running at `http://localhost:5000`.

Intended Use:
-------------
- Local training of the classifier as part of the MLOps pipeline.
- Can be run manually to fine-tune or re-train the ViT model on updated datasets.

Example:
--------
Run directly with Python:

    python train_classifier.py

Dependencies:
-------------
- torch
- torchvision
- transformers
- scikit-learn
- mlflow

"""

import os
import torch
import mlflow
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Lambda, Compose, Resize
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, f1_score

# === Config ===
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MODEL_NAME = "google/vit-base-patch16-224"
DATA_DIR = "data/split"
EXPERIMENT_NAME = "plant-disease-classifier"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    print("⚠️ WARNING: CUDA not available. Training may be slow.")

# === MLflow setup ===
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

# === Processor and transform ===
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

transform_fn = Compose([
    Resize((224, 224)),
    Lambda(lambda img: img.convert("RGB")),
    Lambda(lambda img: processor(images=img, return_tensors="pt")["pixel_values"].squeeze())
])

# === Load datasets ===
train_dataset = ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform_fn)
val_dataset = ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform_fn)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === Model ===
print(f"Loading model for {len(train_dataset.classes)} classes...")
print(f"Classes: {sorted(train_dataset.classes)[:5]}... (showing first 5)")

model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME, num_labels=len(train_dataset.classes), ignore_mismatched_sizes=True
).to(device)

# Verify model has correct output size
actual_outputs = model.classifier.out_features
expected_outputs = len(train_dataset.classes)
print(f"Model classifier output features: {actual_outputs}")
print(f"Expected number of classes: {expected_outputs}")
assert actual_outputs == expected_outputs, f"Model output mismatch! Got {actual_outputs}, expected {expected_outputs}"
print("✓ Model initialized correctly with fresh weights for all classes")


# === Training setup ===
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# === Evaluation function ===
def evaluate(model, loader):
    model.eval()
    total_loss, preds, labels = 0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(pixel_values=x)
            loss = criterion(out.logits, y)
            total_loss += loss.item()
            preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return total_loss / len(loader), acc, f1

# === Training ===
with mlflow.start_run():
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("model", MODEL_NAME)

    mlflow.log_dict(train_dataset.class_to_idx, "class_mapping.json")

    for epoch in range(EPOCHS):
        model.train()
        total_loss, preds, labels = 0, [], []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(pixel_values=x)
            loss = criterion(out.logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")
        val_loss, val_acc, val_f1 = evaluate(model, val_loader)

        print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f} | Train Acc: {acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        mlflow.log_metric("train_loss", total_loss / len(train_loader), step=epoch)
        mlflow.log_metric("train_accuracy", acc, step=epoch)
        mlflow.log_metric("train_f1", f1, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        mlflow.log_metric("val_f1", val_f1, step=epoch)


    num_classes = len(train_dataset.classes)
    num_crops = len(set(c.split("___")[0] for c in train_dataset.classes))
    model_save_path = f"models/vit-finetuned-{num_crops}crops-{num_classes}classes"
    
    print(f"\n{'='*60}")
    print(f"SAVING MODEL TO: {model_save_path}")
    print(f"{'='*60}")
    
    # Save with explicit safe_serialization
    model.save_pretrained(model_save_path, safe_serialization=True)
    processor.save_pretrained(model_save_path)
    
    # Save class mapping
    import json
    class_mapping_path = os.path.join(model_save_path, "class_mapping.json")
    with open(class_mapping_path, 'w') as f:
        json.dump(train_dataset.class_to_idx, f, indent=2)
    
    # VERIFY all required files were saved
    print("\nVerifying saved files...")
    required_files = {
        "config.json": 1000,  # min bytes
        "model.safetensors": 100_000_000,  # min 100MB
        "preprocessor_config.json": 100,
        "class_mapping.json": 100
    }
    
    all_good = True
    for filename, min_size in required_files.items():
        filepath = os.path.join(model_save_path, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            if size >= min_size:
                print(f"✓ {filename}: {size:,} bytes")
            else:
                print(f"✗ {filename}: {size:,} bytes (TOO SMALL, expected >{min_size:,})")
                all_good = False
        else:
            print(f"✗ {filename}: MISSING!")
            all_good = False
    
    if not all_good:
        raise RuntimeError("Model save verification FAILED! Some files are missing or too small.")
    
    print(f"\n✓ All files verified successfully!")
    print(f"Model saved to: {model_save_path}")
    print(f"Classes: {num_classes}, Crops: {num_crops}")
    
    # Log to MLflow
    mlflow.log_artifacts(model_save_path, artifact_path="model")
