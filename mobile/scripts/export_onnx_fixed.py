"""
Re-export ViT model from HuggingFace to ONNX with correct shapes
This should resolve the shape inference issues in the current ONNX model
"""

import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from pathlib import Path
import time

print("=" * 60)
print("ViT Model ONNX Re-Export")
print("=" * 60)
print()

# Paths
model_dir = Path("models/vit-finetuned")
output_onnx = Path("mobile/assets/vit_model_fixed.onnx")

# Check model exists
if not model_dir.exists():
    print(f"❌ Error: Model directory not found at {model_dir}")
    exit(1)

print(f"📦 Loading model from {model_dir}...")
model = ViTForImageClassification.from_pretrained(str(model_dir))
processor = ViTImageProcessor.from_pretrained(str(model_dir))
model.eval()
print("✅ Model loaded successfully")
print()

# Get model info
num_classes = model.config.num_labels
image_size = processor.size['height']
print("Model configuration:")
print(f"  Number of classes: {num_classes}")
print(f"  Image size: {image_size}x{image_size}")
print(f"  Hidden size: {model.config.hidden_size}")
print()

# Create dummy input
print("🔄 Creating dummy input for export...")
dummy_input = torch.randn(1, 3, image_size, image_size)
print(f"  Input shape: {dummy_input.shape}")
print()

# Export to ONNX
print("🚀 Exporting to ONNX...")
print("   This may take 1-2 minutes...")
start_time = time.time()

try:
    torch.onnx.export(
        model,
        dummy_input,
        str(output_onnx),
        input_names=['pixel_values'],
        output_names=['logits'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        opset_version=14,  # Use a stable opset version
        do_constant_folding=True,
        verbose=False,
    )
    
    export_time = time.time() - start_time
    
    # Check output
    if output_onnx.exists():
        output_size = output_onnx.stat().st_size / (1024 * 1024)
        
        print("=" * 60)
        print("✅ SUCCESS!")
        print("=" * 60)
        print(f"📁 Output model: {output_onnx}")
        print(f"   Size: {output_size:.1f} MB")
        print(f"   Export time: {export_time:.1f} seconds")
        print()
        
        # Verify the model
        print("🔍 Verifying exported model...")
        import onnx
        onnx_model = onnx.load(str(output_onnx))
        onnx.checker.check_model(onnx_model)
        print("✅ Model verification passed")
        print()
        
        print("Next steps:")
        print("1. Quantize: python mobile/scripts/quantize_fixed_onnx.py")
        print("2. Verify: python mobile/scripts/verify_onnx_quantization.py")
        print()
        print("Note: The old ONNX files can be kept as backup:")
        print("  - mobile/assets/vit_model.onnx (old)")
        print("  - mobile/assets/vit_model.onnx.data (old)")
        
    else:
        print("❌ Error: Output model was not created")
        exit(1)
        
except Exception as e:
    print(f"❌ Error during export: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
