"""
Simple test: Save entire quantized model object and load it back
"""

import torch
from transformers import ViTForImageClassification
from pathlib import Path
import time

print("Testing: torch.save(entire_model) approach")
print("=" * 60)
print()

# Load and quantize
print("1. Loading and quantizing model...")
model = ViTForImageClassification.from_pretrained("models/vit-finetuned")
model.eval()

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
print("✅ Quantized")
print()

# Save ENTIRE model (not just state_dict)
print("2. Saving entire model object...")
output_path = Path("mobile/assets/vit_quantized_full.pth")
torch.save(quantized_model, str(output_path))
size_mb = output_path.stat().st_size / (1024**2)
print(f"✅ Saved: {size_mb:.1f} MB")
print()

# Load it back
print("3. Loading model back...")
loaded_model = torch.load(str(output_path))
loaded_model.eval()
print("✅ Loaded")
print()

# Test inference
print("4. Testing inference...")
test_input = {"pixel_values": torch.randn(1, 3, 224, 224)}
start = time.time()
with torch.no_grad():
    output = loaded_model(**test_input)
latency = (time.time() - start) * 1000

logits = output.logits[0].numpy()
pred = logits.argmax()

print("✅ Inference works!")
print(f"   Latency: {latency:.1f}ms")
print(f"   Prediction: class {pred}")
print()

print("=" * 60)
print("🎉 SUCCESS! torch.save(model) works perfectly!")
print("=" * 60)
print()
print(f"Quantized model saved to: {output_path}")
print(f"Size: {size_mb:.1f} MB")
print()
print("This model can be loaded and used directly!")
