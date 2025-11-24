"""
Export quantized ViT model to TorchScript for PyTorch Mobile
"""

import torch
from transformers import ViTForImageClassification
from pathlib import Path
import time

print("=" * 70)
print("TorchScript Export - Quantized ViT Model")
print("=" * 70)
print()

# Paths
model_dir = Path("models/vit-finetuned")
output_path = Path("mobile/assets/vit_model_quantized.ptl")

# Step 1: Load and quantize model
print("📦 Step 1: Loading and quantizing model...")
start = time.time()
base_model = ViTForImageClassification.from_pretrained(str(model_dir))
base_model.eval()
print(f"   Model loaded ({time.time() - start:.2f}s)")

print("   Applying INT8 quantization...")
start = time.time()
quantized_base = torch.quantization.quantize_dynamic(
    base_model, {torch.nn.Linear}, dtype=torch.qint8
)
print(f"   Quantization applied ({time.time() - start:.2f}s)")

# Create wrapper to handle tensor input
print("   Creating TorchScript-compatible wrapper...")
class ViTWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, pixel_values):
        # Convert tensor input to dict format
        return self.model(pixel_values=pixel_values)

quantized_model = ViTWrapper(quantized_base)
quantized_model.eval()
print(f"✅ Wrapper created ({time.time() - start:.2f}s)")
print()

# Step 2: Create example input for tracing
print("📸 Step 2: Creating example input...")
# Wrapper expects tensor input directly
example_input = torch.randn(1, 3, 224, 224)
print(f"   Input shape: {example_input.shape}")
print()

# Step 3: Trace the model
print("🔄 Step 3: Tracing model with TorchScript...")
start = time.time()
try:
    # Trace with the dict input
    traced_model = torch.jit.trace(quantized_model, example_input, strict=False)
    print(f"✅ Model traced successfully ({time.time() - start:.2f}s)")
except Exception as e:
    print(f"❌ Tracing failed: {e}")
    print("   This model may not be compatible with TorchScript tracing")
    print("   Trying alternative approach...")
    # Try scripting instead
    try:
        traced_model = torch.jit.script(quantized_model)
        print(f"✅ Model scripted successfully ({time.time() - start:.2f}s)")
    except Exception as e2:
        print(f"❌ Scripting also failed: {e2}")
        print("   Quantized transformers models may not be fully TorchScript compatible")
        exit(1)
print()

# Step 4: Optimize for mobile
print("⚡ Step 4: Optimizing for mobile...")
start = time.time()
try:
    optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(traced_model)
    print(f"✅ Mobile optimization applied ({time.time() - start:.2f}s)")
except Exception as e:
    print(f"⚠️  Mobile optimization failed: {e}")
    print("   Using traced model without mobile optimization")
    optimized_model = traced_model
print()

# Step 5: Save
print("💾 Step 5: Saving TorchScript model...")
start = time.time()
optimized_model._save_for_lite_interpreter(str(output_path))
save_time = time.time() - start
print(f"✅ Model saved ({save_time:.2f}s)")
print()

# Verify
if output_path.exists():
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print("=" * 70)
    print("✅ EXPORT SUCCESSFUL!")
    print("=" * 70)
    print(f"📁 Output: {output_path}")
    print(f"   Size: {size_mb:.1f} MB")
    print()
    print("Next steps:")
    print("1. Test the .ptl file: python mobile/scripts/test_torchscript.py")
    print("2. Run validation on full dataset")
    print("3. Deploy to Android app")
else:
    print("❌ Error: Output file not created")
    exit(1)
