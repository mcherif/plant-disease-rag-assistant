"""
evaluate_test.py

This script defines a pytest-based integration test for evaluating a fine-tuned
Vision Transformer (ViT) model on a plant disease classification dataset.

Key Features:
-------------
- Loads a saved Hugging Face ViT model and its associated processor.
- Applies consistent image preprocessing using the same processor.
- Loads the test dataset and applies class label remapping.
- Evaluates the model using accuracy and weighted F1-score.
- Logs evaluation metrics to MLflow for experiment tracking.
- Skips the test automatically in CI environments if the model checkpoint
  (pytorch_model.bin) is missing, to prevent unnecessary failures.

Intended Use:
-------------
- Run as part of your CI/CD test suite to validate that the trained model can
  be correctly loaded and evaluated.
- Can be executed manually with `pytest src/training/evaluate_test.py`.

Requirements:
-------------
- A trained model saved in `models/vit-finetuned/`
- A test dataset in `data/split/test/`
- MLflow tracking server (or CI-compatible fallback using local filesystem)

"""


import os
import json
import torch
import mlflow
import pytest
import tempfile
from collections import Counter
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, f1_score

try:
    from torchvision.datasets import ImageFolder
    from torchvision.transforms import Lambda, Compose, Resize
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

# === Config ===
MODEL_DIR = "models/vit-finetuned"
DATA_DIR = "data/split/test"
BATCH_SIZE = 16
EXPERIMENT_NAME = "plant-disease-classifier"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin")),
    reason="Model not found — skipping evaluation test in CI"
)
@pytest.mark.skipif(
    not TORCHVISION_AVAILABLE,
    reason="torchvision not installed"
)
def test_model_evaluation_pipeline():
    # Setup MLflow tracking URI
    if os.environ.get("CI", "false").lower() == "true":
        mlflow.set_tracking_uri(f"file://{tempfile.gettempdir()}/mlruns")
    else:
        mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load model and processor
    model = AutoModelForImageClassification.from_pretrained(MODEL_DIR).to(device)
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)

    # Prepare transform
    transform_fn = Compose([
        Resize((224, 224)),
        Lambda(lambda img: img.convert("RGB")),
        Lambda(lambda img: processor(images=img, return_tensors="pt")["pixel_values"].squeeze())
    ])

    # Load dataset
    with open(os.path.join(MODEL_DIR, "class_mapping.json")) as f:
        class_mapping = json.load(f)

    test_dataset = ImageFolder(DATA_DIR, transform=transform_fn)
    test_dataset.class_to_idx = class_mapping
    test_dataset.classes = [k for k, _ in sorted(class_mapping.items(), key=lambda x: x[1])]
    index_remap = dict((i, class_mapping[class_name]) for i, class_name in enumerate(test_dataset.classes))
    test_dataset.targets = [index_remap[label] for label in test_dataset.targets]

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Evaluation function
    def evaluate(model, loader):
        model.eval()
        loss_fn = torch.nn.CrossEntropyLoss()
        preds, labels = [], []
        total_loss = 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(pixel_values=x)
                loss = loss_fn(out.logits, y)
                total_loss += loss.item()
                preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
                labels.extend(y.cpu().numpy())

        print("✅ True label distribution:", Counter(labels))
        print("🔍 Predicted label distribution:", Counter(preds))
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")
        return total_loss / len(loader), acc, f1

    # Run test eval and log to MLflow
    with mlflow.start_run(run_name="test_evaluation_fixed_resize"):
        test_loss, test_acc, test_f1 = evaluate(model, test_loader)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1", test_f1)
        print(f"\n🧪 Test Evaluation → Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
