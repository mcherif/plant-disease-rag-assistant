"""
ONNX Quantization using existing working ONNX model
Skip shape inference and quantize directly
"""

from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path
import time

print("=" * 70)
print("ONNX Dynamic Quantization (Using Existing Model)")
print("=" * 70)
print()

# Paths
input_onnx = Path("mobile/assets/vit_model_fixed.onnx")
output_onnx = Path("mobile/assets/vit_model_int8_simple.onnx")

# Check input exists
if not input_onnx.exists():
    print(f"❌ Error: Input ONNX model not found at {input_onnx}")
    print("   Please run: python mobile/scripts/export_onnx_fixed.py first")
    exit(1)

input_size = input_onnx.stat().st_size / (1024 * 1024)
print(f"📁 Input model: {input_onnx}")
print(f"   Size: {input_size:.1f} MB")

# Check for external data
external_data = input_onnx.with_suffix('.onnx.data')
if external_data.exists():
    ext_size = external_data.stat().st_size / (1024 * 1024)
    print(f"   External data: {ext_size:.1f} MB")
    print(f"   Total: {input_size + ext_size:.1f} MB")
print()

# Quantize directly (skip shape inference)
print("⚡ Quantizing to INT8...")
print("   This may take 1-2 minutes...")
start = time.time()

try:
    quantize_dynamic(
        str(input_onnx),
        str(output_onnx),
        weight_type=QuantType.QUInt8  # Try QUInt8 instead of QInt8
    )
    
    quant_time = time.time() - start
    print(f"✅ Quantization complete ({quant_time:.1f}s)")
    print()
    
    # Check output
    if output_onnx.exists():
        output_size = output_onnx.stat().st_size / (1024 * 1024)
        print(f"📁 Output model: {output_onnx}")
        print(f"   Size: {output_size:.1f} MB")
        
        # Check for external data
        output_ext = output_onnx.with_suffix('.onnx.data')
        if output_ext.exists():
            out_ext_size = output_ext.stat().st_size / (1024 * 1024)
            print(f"   External data: {out_ext_size:.1f} MB")
            total_output = output_size + out_ext_size
            print(f"   Total: {total_output:.1f} MB")
            
            # Calculate compression
            total_input = input_size
            if external_data.exists():
                total_input += ext_size
            compression = (1 - total_output / total_input) * 100
            print(f"   Compression: {compression:.1f}% smaller")
        
        print()
        print("=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("1. Test: python mobile/scripts/test_onnx_quantized.py")
        print("2. Deploy to Android with ONNX Runtime Mobile")
        
    else:
        print("❌ Error: Output model not created")
        exit(1)
        
except Exception as e:
    print(f"❌ Quantization failed: {e}")
    import traceback
    traceback.print_exc()
    print()
    print("This is the same shape inference error we encountered before.")
    print("The ONNX model has fundamental shape issues that prevent quantization.")
    exit(1)
