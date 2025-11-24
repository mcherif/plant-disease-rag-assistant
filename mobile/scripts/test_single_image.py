"""
Single image test with verbose logging to debug quantized model
"""

import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from pathlib import Path
from PIL import Image
import numpy as np
import time

print("=" * 70)
print("Single Image Quantized Model Test (Verbose)")
print("=" * 70)
print()

# Paths
model_dir = Path("models/vit-finetuned")
test_image = Path("data/sample.jpg")

if not test_image.exists():
    # Find any validation image
    val_dir = Path("data/split/val")
    for class_dir in val_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.JPG"))
            if images:
                test_image = images[0]
                break

print(f"Test image: {test_image}")
print()

# Load original model
print("📦 Step 1: Loading original FP32 model...")
start = time.time()
original_model = ViTForImageClassification.from_pretrained(str(model_dir))
processor = ViTImageProcessor.from_pretrained(str(model_dir))
original_model.eval()
print(f"✅ Original model loaded ({time.time() - start:.2f}s)")
print()

# Load and preprocess image
print("📸 Step 2: Loading and preprocessing image...")
start = time.time()
image = Image.open(test_image).convert("RGB")
inputs = processor(images=image, return_tensors="pt")
print(f"✅ Image preprocessed ({time.time() - start:.2f}s)")
print(f"   Input shape: {inputs['pixel_values'].shape}")
print()

# Run original model
print("🔄 Step 3: Running FP32 model inference...")
start = time.time()
with torch.no_grad():
    original_outputs = original_model(**inputs)
original_latency = time.time() - start
original_logits = original_outputs.logits[0].numpy()
original_pred = original_logits.argmax()
print(f"✅ FP32 inference complete ({original_latency*1000:.1f}ms)")
print(f"   Top prediction: class {original_pred}")
print(f"   Top-3: {np.argsort(original_logits)[-3:][::-1].tolist()}")
print()

# Create quantized model
print("📦 Step 4: Creating quantized INT8 model...")
start = time.time()
quantized_model = ViTForImageClassification.from_pretrained(str(model_dir))
quantized_model.eval()
print(f"   Model loaded ({time.time() - start:.2f}s)")

print("   Applying dynamic quantization...")
start = time.time()
quantized_model = torch.quantization.quantize_dynamic(
    quantized_model, {torch.nn.Linear}, dtype=torch.qint8
)
print(f"✅ Quantization applied ({time.time() - start:.2f}s)")
print()

# Run quantized model
print("🔄 Step 5: Running INT8 model inference...")
start = time.time()
with torch.no_grad():
    quantized_outputs = quantized_model(**inputs)
quantized_latency = time.time() - start
quantized_logits = quantized_outputs.logits[0].numpy()
quantized_pred = quantized_logits.argmax()
print(f"✅ INT8 inference complete ({quantized_latency*1000:.1f}ms)")
print(f"   Top prediction: class {quantized_pred}")
print(f"   Top-3: {np.argsort(quantized_logits)[-3:][::-1].tolist()}")
print()

# Compare results
print("=" * 70)
print("📊 COMPARISON")
print("=" * 70)
print()
print(f"Predictions match: {'✅ Yes' if original_pred == quantized_pred else '❌ No'}")
print(f"FP32 latency: {original_latency*1000:.1f}ms")
print(f"INT8 latency: {quantized_latency*1000:.1f}ms")
print(f"Speedup: {original_latency/quantized_latency:.2f}x")
print()

# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarity = cosine_similarity(original_logits, quantized_logits)
print(f"Output similarity: {similarity:.4f}")
print()

if similarity > 0.95 and original_pred == quantized_pred:
    print("✅ QUANTIZATION WORKING CORRECTLY!")
else:
    print("⚠️  Some differences detected")
print("=" * 70)
