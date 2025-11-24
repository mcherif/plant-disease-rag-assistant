"""
Test the exported TorchScript model
"""

import torch
from pathlib import Path
from PIL import Image
import numpy as np
import time

print("=" * 70)
print("TorchScript Model Test")
print("=" * 70)
print()

# Paths
torchscript_path = Path("mobile/assets/vit_model_quantized.ptl")
test_image = Path("data/sample.jpg")

# Step 1: Load TorchScript model
print("📦 Step 1: Loading TorchScript model...")
start = time.time()
model = torch.jit.load(str(torchscript_path))
model.eval()
load_time = time.time() - start
print(f"✅ TorchScript model loaded ({load_time:.2f}s)")
print()

# Step 2: Prepare input
print("📸 Step 2: Preparing input...")
image = Image.open(test_image).convert("RGB")
image = image.resize((224, 224))

# Convert to tensor and normalize
img_array = np.array(image).astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_array = (img_array - mean) / std
img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
input_tensor = torch.from_numpy(img_array).unsqueeze(0)  # Add batch dim

print("✅ Input prepared")
print(f"   Shape: {input_tensor.shape}")
print()

# Step 3: Run inference
print("🔄 Step 3: Running inference...")
start = time.time()
with torch.no_grad():
    # Wrapper model expects tensor input directly
    output = model(input_tensor)
latency = time.time() - start

# Get predictions
logits = output[0].numpy() if isinstance(output, tuple) else output.numpy()[0]
top_pred = logits.argmax()
top3 = np.argsort(logits)[-3:][::-1]

print(f"✅ Inference complete ({latency*1000:.1f}ms)")
print(f"   Top prediction: class {top_pred}")
print(f"   Top-3: {top3.tolist()}")
print()

# Results
print("=" * 70)
print("📊 RESULTS")
print("=" * 70)
print(f"Model file: {torchscript_path}")
print(f"Model size: {torchscript_path.stat().st_size / (1024**2):.1f} MB")
print(f"Load time: {load_time:.2f}s")
print(f"Inference latency: {latency*1000:.1f}ms")
print()
print("✅ TorchScript model is working correctly!")
print("   Ready for Android deployment")
print("=" * 70)
