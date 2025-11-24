"""
Test model predictions on actual training images
"""
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import json
from pathlib import Path

MODEL_DIR = "models/vit-finetuned-15crops-41classes"

# Load model and processor
processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
model.eval()

# Load class mapping
with open(f"{MODEL_DIR}/class_mapping.json") as f:
    class_mapping = json.load(f)
    idx_to_class = {v: k for k, v in class_mapping.items()}

def test_image(image_path, expected_class):
    img = Image.open(image_path).convert('RGB')
    print(f"\nTesting: {image_path.name}")
    print(f"  Original size: {img.size}")
    print(f"  Expected class: {expected_class}")
    
    # Preprocess
    inputs = processor(images=img, return_tensors="pt")
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        top_5 = torch.topk(probs, 5)
    
    print("  Top 5 predictions:")
    for prob, idx in zip(top_5.values, top_5.indices):
        class_name = idx_to_class[idx.item()]
        print(f"    {class_name}: {prob.item():.4f}")
    
    top_pred = idx_to_class[top_5.indices[0].item()]
    if expected_class in top_pred:
        print("  ✅ CORRECT!")
    else:
        print(f"  ❌ WRONG! (expected {expected_class})")

# Test Olive images from training set
print("="*60)
print("TESTING OLIVE IMAGES FROM TRAINING SET")
print("="*60)

olive_healthy_dir = Path("data/split/train/Olive___healthy")
olive_peacock_dir = Path("data/split/train/Olive___Peacock_spot")
olive_aculus_dir = Path("data/split/train/Olive___Aculus_olearius")

# Test first 3 images from each class
print("\n--- Olive Healthy ---")
for img in list(olive_healthy_dir.glob("*.jpg"))[:3]:
    test_image(img, "Olive___healthy")

print("\n--- Olive Peacock Spot ---")
for img in list(olive_peacock_dir.glob("*.jpg"))[:3]:
    test_image(img, "Olive___Peacock_spot")

print("\n--- Olive Aculus olearius ---")
for img in list(olive_aculus_dir.glob("*.jpg"))[:3]:
    test_image(img, "Olive___Aculus_olearius")

# Compare with Peach
print("\n\n" + "="*60)
print("TESTING PEACH IMAGES (for comparison)")
print("="*60)

peach_dir = Path("data/split/train/Peach___Bacterial_spot")
print("\n--- Peach Bacterial Spot ---")
for img in list(peach_dir.glob("*.jpg"))[:3]:
    test_image(img, "Peach___Bacterial_spot")
