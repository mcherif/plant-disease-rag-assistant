"""
Quick verification of PyTorch quantized model on a small sample
Tests on 20 images to get fast feedback
"""

import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from pathlib import Path
from PIL import Image
import numpy as np
import json
import time
import random

print("=" * 70)
print("Quick PyTorch Quantized Model Verification (Sample)")
print("=" * 70)
print()

# Paths
model_dir = Path("models/vit-finetuned")
val_dir = Path("data/split/val")

# Load models
print("📦 Loading models...")
original_model = ViTForImageClassification.from_pretrained(str(model_dir))
processor = ViTImageProcessor.from_pretrained(str(model_dir))
original_model.eval()

quantized_model = ViTForImageClassification.from_pretrained(str(model_dir))
quantized_model.eval()
quantized_model = torch.quantization.quantize_dynamic(
    quantized_model, {torch.nn.Linear}, dtype=torch.qint8
)
print("✅ Models loaded")
print()

# Get class mapping
with open(model_dir / "class_mapping.json", "r") as f:
    class_mapping = json.load(f)

# Collect sample images (50 images, ~1-2 per class)
print("📁 Collecting sample images...")
val_images = []
val_labels = []

for class_idx, (class_name, _) in enumerate(class_mapping.items()):
    class_dir = val_dir / class_name
    if not class_dir.exists():
        continue
    
    images = list(class_dir.glob("*.JPG"))
    # Take 1-2 images per class
    sample = random.sample(images, min(2, len(images)))
    for img_path in sample:
        val_images.append(img_path)
        val_labels.append(class_idx)

# Limit to 20 images total
if len(val_images) > 20:
    indices = random.sample(range(len(val_images)), 20)
    val_images = [val_images[i] for i in indices]
    val_labels = [val_labels[i] for i in indices]

print(f"✅ Testing on {len(val_images)} sample images")
print()

# Run verification
print("🔄 Running verification...")
original_correct = 0
quantized_correct = 0
top3_original_correct = 0
top3_quantized_correct = 0
latencies_original = []
latencies_quantized = []
cosine_similarities = []

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

with torch.no_grad():
    for i, (img_path, true_label) in enumerate(zip(val_images, val_labels)):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(val_images)} images...")
        
        # Load and preprocess
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        
        # Original model
        start = time.time()
        original_outputs = original_model(**inputs)
        latencies_original.append(time.time() - start)
        original_logits = original_outputs.logits[0].numpy()
        original_pred = original_logits.argmax()
        original_top3 = np.argsort(original_logits)[-3:][::-1]
        
        # Quantized model
        start = time.time()
        quantized_outputs = quantized_model(**inputs)
        latencies_quantized.append(time.time() - start)
        quantized_logits = quantized_outputs.logits[0].numpy()
        quantized_pred = quantized_logits.argmax()
        quantized_top3 = np.argsort(quantized_logits)[-3:][::-1]
        
        # Metrics
        if original_pred == true_label:
            original_correct += 1
        if quantized_pred == true_label:
            quantized_correct += 1
        if true_label in original_top3:
            top3_original_correct += 1
        if true_label in quantized_top3:
            top3_quantized_correct += 1
        
        similarity = cosine_similarity(original_logits, quantized_logits)
        cosine_similarities.append(similarity)

print()

# Results
total = len(val_images)
original_top1_acc = (original_correct / total) * 100
quantized_top1_acc = (quantized_correct / total) * 100
original_top3_acc = (top3_original_correct / total) * 100
quantized_top3_acc = (top3_quantized_correct / total) * 100
avg_similarity = np.mean(cosine_similarities)
avg_latency_original = np.mean(latencies_original) * 1000
avg_latency_quantized = np.mean(latencies_quantized) * 1000
speedup = avg_latency_original / avg_latency_quantized

print("=" * 70)
print("📊 QUICK VERIFICATION RESULTS (Sample)")
print("=" * 70)
print()
print(f"Sample size: {total} images")
print()
print("Accuracy Metrics:")
print(f"  Original FP32 Top-1: {original_top1_acc:.1f}%")
print(f"  Quantized INT8 Top-1: {quantized_top1_acc:.1f}%")
print(f"  Accuracy loss: {original_top1_acc - quantized_top1_acc:.1f}%")
print()
print(f"  Original FP32 Top-3: {original_top3_acc:.1f}%")
print(f"  Quantized INT8 Top-3: {quantized_top3_acc:.1f}%")
print(f"  Accuracy loss: {original_top3_acc - quantized_top3_acc:.1f}%")
print()
print("Inference Latency:")
print(f"  Original FP32: {avg_latency_original:.1f} ms")
print(f"  Quantized INT8: {avg_latency_quantized:.1f} ms")
print(f"  Speedup: {speedup:.2f}x")
print()
print("Output Similarity:")
print(f"  Avg cosine similarity: {avg_similarity:.4f}")
print(f"  Min similarity: {min(cosine_similarities):.4f}")
print(f"  Max similarity: {max(cosine_similarities):.4f}")
print()

# Verdict
print("=" * 70)
if quantized_top1_acc >= original_top1_acc - 5.0 and avg_similarity > 0.95:
    print("✅ SAMPLE VERIFICATION SUCCESSFUL!")
    print("   Quantization looks good. Full validation recommended.")
elif quantized_top1_acc >= original_top1_acc - 10.0 and avg_similarity > 0.90:
    print("⚠️  SAMPLE VERIFICATION ACCEPTABLE")
    print("   Some accuracy loss. Full validation needed.")
else:
    print("❌ SAMPLE VERIFICATION CONCERNING")
    print("   Significant issues detected. Review needed.")
print("=" * 70)
print()
print("Note: This is a small sample. Run full validation for final results.")
