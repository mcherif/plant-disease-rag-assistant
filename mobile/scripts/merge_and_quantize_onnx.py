"""
Merge ONNX external data into single file, then quantize
This resolves issues with ONNX Runtime quantization on models with external data
"""

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path
import time

print("=" * 60)
print("ONNX External Data Merge + Quantization")
print("=" * 60)
print()

# Paths
input_model = Path("mobile/assets/vit_model.onnx")
input_data = Path("mobile/assets/vit_model.onnx.data")
merged_model = Path("mobile/assets/vit_model_merged.onnx")
output_model = Path("mobile/assets/vit_model_int8.onnx")

# Check input files exist
if not input_model.exists():
    print(f"❌ Error: Input model not found at {input_model}")
    exit(1)

if not input_data.exists():
    print(f"❌ Error: External data not found at {input_data}")
    exit(1)

input_size = (input_model.stat().st_size + input_data.stat().st_size) / (1024 * 1024)
print(f"📁 Input model: {input_model}")
print(f"📁 External data: {input_data}")
print(f"   Total size: {input_size:.1f} MB")
print()

# Step 1: Merge external data
print("🔄 Step 1: Merging external data into single file...")
print("   This may take 30-60 seconds...")
start_time = time.time()

try:
    # Load model with external data
    model = onnx.load(str(input_model), load_external_data=True)
    
    # Save as single file
    onnx.save(model, str(merged_model))
    
    merge_time = time.time() - start_time
    merged_size = merged_model.stat().st_size / (1024 * 1024)
    
    print(f"✅ Merged model created: {merged_model}")
    print(f"   Size: {merged_size:.1f} MB")
    print(f"   Time: {merge_time:.1f} seconds")
    print()
    
except Exception as e:
    print(f"❌ Error during merge: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 2: Quantize merged model
print("🔄 Step 2: Quantizing merged model to INT8...")
print("   This may take 1-2 minutes...")
quant_start = time.time()

try:
    quantize_dynamic(
        model_input=str(merged_model),
        model_output=str(output_model),
        weight_type=QuantType.QUInt8,
    )
    
    quant_time = time.time() - quant_start
    
    # Check output
    if output_model.exists():
        output_size = output_model.stat().st_size / (1024 * 1024)
        compression_ratio = (1 - output_size / input_size) * 100
        total_time = time.time() - start_time
        
        print("=" * 60)
        print("✅ SUCCESS!")
        print("=" * 60)
        print(f"📁 Output model: {output_model}")
        print(f"   Size: {output_size:.1f} MB")
        print(f"   Compression: {compression_ratio:.1f}% smaller")
        print(f"   Merge time: {merge_time:.1f}s")
        print(f"   Quantization time: {quant_time:.1f}s")
        print(f"   Total time: {total_time:.1f}s")
        print()
        print("Next steps:")
        print("1. Run verification: python mobile/scripts/verify_onnx_quantization.py")
        print("2. Test on Android device")
        print()
        print("Note: You can delete the merged model to save space:")
        print(f"   del {merged_model}")
    else:
        print("❌ Error: Output model was not created")
        exit(1)
        
except Exception as e:
    print(f"❌ Error during quantization: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
