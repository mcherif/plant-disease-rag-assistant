"""
ONNX Runtime Dynamic Quantization Script
Quantizes the ViT ONNX model to INT8 for reduced size and faster inference.
"""

from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path
import time

print("=" * 60)
print("ONNX Runtime Dynamic Quantization")
print("=" * 60)
print()

# Paths
input_model = Path("mobile/assets/vit_model.onnx")
output_model = Path("mobile/assets/vit_model_int8.onnx")

# Check input model exists
if not input_model.exists():
    print(f"❌ Error: Input model not found at {input_model}")
    exit(1)

input_size = input_model.stat().st_size / (1024 * 1024)
print(f"📁 Input model: {input_model}")
print(f"   Size: {input_size:.1f} MB")
print()

# Quantize to INT8
print("🚀 Starting dynamic quantization to INT8...")
print("   This may take 1-2 minutes...")
print()

start_time = time.time()

try:
    quantize_dynamic(
        model_input=str(input_model),
        model_output=str(output_model),
        weight_type=QuantType.QUInt8,  # Unsigned INT8
    )
    
    elapsed = time.time() - start_time
    
    # Check output
    if output_model.exists():
        output_size = output_model.stat().st_size / (1024 * 1024)
        compression_ratio = (1 - output_size / input_size) * 100
        
        print("=" * 60)
        print("✅ SUCCESS!")
        print("=" * 60)
        print(f"📁 Output model: {output_model}")
        print(f"   Size: {output_size:.1f} MB")
        print(f"   Compression: {compression_ratio:.1f}% smaller")
        print(f"   Time: {elapsed:.1f} seconds")
        print()
        print("Next steps:")
        print("1. Run verification: python mobile/scripts/verify_onnx_quantization.py")
        print("2. Test on Android device")
    else:
        print("❌ Error: Output model was not created")
        exit(1)
        
except Exception as e:
    print(f"❌ Error during quantization: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
