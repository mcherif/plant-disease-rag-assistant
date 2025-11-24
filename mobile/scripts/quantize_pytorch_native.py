"""
PyTorch Native Dynamic Quantization
Quantizes the model directly in PyTorch, then exports to ONNX
This bypasses ONNX Runtime's shape inference issues
"""

import torch
from transformers import ViTForImageClassification
from pathlib import Path
import time

print("=" * 60)
print("PyTorch Native Dynamic Quantization")
print("=" * 60)
print()

# Paths
model_dir = Path("models/vit-finetuned")
output_path = Path("mobile/assets/vit_model_quantized.pth")

# Load model
print(f"📦 Loading model from {model_dir}...")
model = ViTForImageClassification.from_pretrained(str(model_dir))
model.eval()
print("✅ Model loaded successfully")
print()

# Get model size before quantization
def get_model_size(model):
    import tempfile
    import os
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
    temp_path = temp_file.name
    temp_file.close()
    
    torch.save(model.state_dict(), temp_path)
    size = Path(temp_path).stat().st_size / (1024 * 1024)
    
    try:
        os.remove(temp_path)
    except Exception:
        pass  # Ignore cleanup errors
    
    return size

original_size = get_model_size(model)
print(f"Original model size: {original_size:.1f} MB")
print()

# Quantize
print("🚀 Applying dynamic quantization...")
print("   Quantizing Linear layers to INT8...")
start_time = time.time()

quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Quantize all Linear layers
    dtype=torch.qint8
)

quant_time = time.time() - start_time
print(f"✅ Quantization complete ({quant_time:.1f}s)")
print()

# Save quantized model
print("💾 Saving quantized model...")
torch.save(quantized_model.state_dict(), str(output_path))
quantized_size = output_path.stat().st_size / (1024 * 1024)

print("=" * 60)
print("✅ SUCCESS!")
print("=" * 60)
print(f"📁 Output model: {output_path}")
print(f"   Original size: {original_size:.1f} MB")
print(f"   Quantized size: {quantized_size:.1f} MB")
print(f"   Compression: {(1 - quantized_size/original_size)*100:.1f}% smaller")
print()
print("Next steps:")
print("1. Test quantized model: python mobile/scripts/test_pytorch_quantized.py")
print("2. Export to ONNX if needed (quantization preserved)")
print()
print("Note: This is a PyTorch model (.pth), not ONNX")
print("For Android, you can use PyTorch Mobile or export to ONNX")
