"""
Verification script for ONNX quantized model
Compares FP32 vs INT8 model outputs and performance
"""

import onnxruntime as ort
import numpy as np
from pathlib import Path
from PIL import Image
import time

print("=" * 60)
print("ONNX Quantization Verification")
print("=" * 60)
print()

# Paths
fp32_model = Path("mobile/assets/vit_model.onnx")
int8_model = Path("mobile/assets/vit_model_int8.onnx")
sample_image = Path("data/sample.jpg")

# Check files exist
for path in [fp32_model, int8_model, sample_image]:
    if not path.exists():
        print(f"❌ Error: {path} not found")
        exit(1)

# Preprocessing function
def preprocess_image(img_path):
    """Preprocess image for ViT model"""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    
    # Convert to (C, H, W) and add batch dimension
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, axis=0).astype(np.float32)

# Load models
print("📦 Loading models...")
fp32_session = ort.InferenceSession(str(fp32_model))
int8_session = ort.InferenceSession(str(int8_model))
print("✅ Models loaded")
print()

# Get input name
input_name = fp32_session.get_inputs()[0].name

# Preprocess image
print(f"📸 Processing image: {sample_image}")
input_data = preprocess_image(sample_image)
print(f"   Input shape: {input_data.shape}")
print()

# Run FP32 inference
print("🔄 Running FP32 inference...")
fp32_times = []
for i in range(5):
    start = time.time()
    fp32_output = fp32_session.run(None, {input_name: input_data})[0]
    fp32_times.append(time.time() - start)

fp32_latency = np.mean(fp32_times[1:]) * 1000  # Skip first run, convert to ms
print(f"   Latency: {fp32_latency:.1f} ms (avg of 4 runs)")
print(f"   Output shape: {fp32_output.shape}")
print()

# Run INT8 inference
print("🔄 Running INT8 inference...")
int8_times = []
for i in range(5):
    start = time.time()
    int8_output = int8_session.run(None, {input_name: input_data})[0]
    int8_times.append(time.time() - start)

int8_latency = np.mean(int8_times[1:]) * 1000  # Skip first run, convert to ms
print(f"   Latency: {int8_latency:.1f} ms (avg of 4 runs)")
print(f"   Output shape: {int8_output.shape}")
print()

# Calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a.flatten(), b.flatten()) / (
        np.linalg.norm(a.flatten()) * np.linalg.norm(b.flatten())
    )

similarity = cosine_similarity(fp32_output, int8_output)

# Get top predictions
fp32_top3 = np.argsort(fp32_output[0])[-3:][::-1]
int8_top3 = np.argsort(int8_output[0])[-3:][::-1]

# Results
print("=" * 60)
print("📊 RESULTS")
print("=" * 60)
print()
print("Model Sizes:")
print(f"  FP32: {fp32_model.stat().st_size / (1024**2):.1f} MB")
print(f"  INT8: {int8_model.stat().st_size / (1024**2):.1f} MB")
print()
print("Inference Latency:")
print(f"  FP32: {fp32_latency:.1f} ms")
print(f"  INT8: {int8_latency:.1f} ms")
print(f"  Speedup: {fp32_latency / int8_latency:.2f}x")
print()
print("Output Similarity:")
print(f"  Cosine similarity: {similarity:.4f}")
print()
print("Top-3 Predictions:")
print(f"  FP32: {fp32_top3.tolist()}")
print(f"  INT8: {int8_top3.tolist()}")
print(f"  Match: {'✅ Yes' if np.array_equal(fp32_top3, int8_top3) else '⚠️  No'}")
print()

# Verdict
if similarity > 0.95 and np.array_equal(fp32_top3, int8_top3):
    print("✅ QUANTIZATION SUCCESSFUL!")
    print("   The INT8 model is ready for deployment.")
elif similarity > 0.90:
    print("⚠️  QUANTIZATION ACCEPTABLE")
    print("   Minor differences detected, but likely acceptable.")
else:
    print("❌ QUANTIZATION FAILED")
    print("   Significant accuracy loss detected.")
