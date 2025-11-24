"""
Static Quantization for ViT Model
Quantizes both weights AND activations for better TorchScript compatibility
"""

import time
from pathlib import Path

import torch
from PIL import Image
from torch.quantization import convert, get_default_qconfig, prepare
from transformers import ViTForImageClassification, ViTImageProcessor

print("=" * 70)
print("Static Quantization - ViT Model")
print("=" * 70)
print()

# Paths
model_dir = Path("models/vit-finetuned")
output_path = Path("mobile/assets/vit_model_static_int8.ptl")
calib_dir = Path("mobile/data/calibration_images")

# Step 1: Load model
print("📦 Step 1: Loading FP32 model...")
start = time.time()
model = ViTForImageClassification.from_pretrained(str(model_dir))
processor = ViTImageProcessor.from_pretrained(str(model_dir))
model.eval()
print(f"✅ Model loaded ({time.time() - start:.2f}s)")
print()

# Step 2: Set quantization config
print("⚙️  Step 2: Configuring static quantization...")
# Use qnnpack backend (optimized for mobile)
model.qconfig = get_default_qconfig('qnnpack')
print("✅ Quantization config set (qnnpack backend)")
print()

# Step 3: Prepare model for calibration
print("🔧 Step 3: Preparing model for calibration...")
start = time.time()
model_prepared = prepare(model, inplace=False)
print(f"✅ Model prepared ({time.time() - start:.2f}s)")
print()

# Step 4: Calibrate with representative data
print("📊 Step 4: Calibrating with representative images...")
calib_images = list(calib_dir.glob("*.JPG"))[:10]
print(f"   Using {len(calib_images)} calibration images")

start = time.time()
with torch.no_grad():
    for i, img_path in enumerate(calib_images):
        print(f"   Processing {i+1}/{len(calib_images)}...", end='\r')
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        _ = model_prepared(**inputs)

calib_time = time.time() - start
print(f"\n✅ Calibration complete ({calib_time:.2f}s)")
print()

# Step 5: Convert to quantized model
print("🔄 Step 5: Converting to quantized model...")
start = time.time()
quantized_model = convert(model_prepared, inplace=False)
print(f"✅ Conversion complete ({time.time() - start:.2f}s)")
print()

# Step 6: Create wrapper for TorchScript
print("📦 Step 6: Creating TorchScript wrapper...")
class ViTWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values)

wrapped_model = ViTWrapper(quantized_model)
wrapped_model.eval()
print("✅ Wrapper created")
print()

# Step 7: Export to TorchScript
print("🚀 Step 7: Exporting to TorchScript...")
example_input = torch.randn(1, 3, 224, 224)

start = time.time()
try:
    traced = torch.jit.trace(wrapped_model, example_input, strict=False)
    print(f"✅ Model traced ({time.time() - start:.2f}s)")
    
    # Optimize for mobile
    print("   Optimizing for mobile...")
    optimized = torch.utils.mobile_optimizer.optimize_for_mobile(traced)
    
    # Save
    optimized._save_for_lite_interpreter(str(output_path))
    
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print()
        print("=" * 70)
        print("✅ STATIC QUANTIZATION SUCCESSFUL!")
        print("=" * 70)
        print(f"📁 Output: {output_path}")
        print(f"   Size: {size_mb:.1f} MB")
        print()
        print("Next: Test with python mobile/scripts/test_torchscript.py")
    else:
        print("❌ Error: Output file not created")
        exit(1)
        
except Exception as e:
    print(f"❌ TorchScript export failed: {e}")
    print()
    print("Static quantization may not be compatible with this model.")
    print("Recommendation: Try TensorFlow Lite conversion next.")
    exit(1)
