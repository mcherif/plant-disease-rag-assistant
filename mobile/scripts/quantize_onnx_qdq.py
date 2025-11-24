"""
ONNX QDQ Quantization for Mobile - Recommended by Alternative LLM
Pipeline: Clean ONNX export → Shape inference → QDQ quantization → ORT format
"""

import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from pathlib import Path
from PIL import Image
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import time

print("=" * 70)
print("ONNX QDQ Quantization for Mobile")
print("=" * 70)
print()

# Paths
model_dir = Path("models/vit-finetuned")
sample_image = Path("data/sample.jpg")
onnx_fp32 = Path("mobile/assets/vit_fp32_clean.onnx")
onnx_inferred = Path("mobile/assets/vit_fp32_inferred.onnx")
onnx_int8 = Path("mobile/assets/vit_int8_qdq.onnx")

# Step 1: Load model
print("📦 Step 1: Loading model...")
model = ViTForImageClassification.from_pretrained(str(model_dir), torchscript=False)
processor = ViTImageProcessor.from_pretrained(str(model_dir))
model.eval()
print("✅ Model loaded")
print()

# Step 2: Create proper dummy input
print("📸 Step 2: Creating dummy input with correct shape...")
if sample_image.exists():
    image = Image.open(sample_image)
    dummy_input = processor(images=image, return_tensors="pt")["pixel_values"]
else:
    # Fallback to random tensor
    dummy_input = torch.randn(1, 3, 224, 224)
print(f"   Input shape: {dummy_input.shape}")
print()

# Step 3: Export to ONNX with stable shapes
print("🚀 Step 3: Exporting to ONNX (opset 17, external data)...")
start = time.time()

try:
    torch.onnx.export(
        model,
        (dummy_input,),
        str(onnx_fp32),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "logits": {0: "batch"}
        },
        opset_version=17,
        do_constant_folding=True
    )
    
    export_time = time.time() - start
    print(f"✅ ONNX export complete ({export_time:.1f}s)")
    
    if onnx_fp32.exists():
        size_mb = onnx_fp32.stat().st_size / (1024 * 1024)
        print(f"   Model file: {size_mb:.1f} MB")
        
        # Check for external data file
        external_data = onnx_fp32.with_suffix('.onnx.data')
        if external_data.exists():
            ext_size_mb = external_data.stat().st_size / (1024 * 1024)
            print(f"   External data: {ext_size_mb:.1f} MB")
            print(f"   Total: {size_mb + ext_size_mb:.1f} MB")
    else:
        print("❌ Error: ONNX file not created")
        exit(1)
        
except Exception as e:
    print(f"❌ ONNX export failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# Step 4: Run ONNX checker and shape inference
print("🔍 Step 4: Running ONNX checker and shape inference...")
start = time.time()

try:
    # Load and check model
    onnx_model = onnx.load(str(onnx_fp32))
    onnx.checker.check_model(onnx_model)
    print("   ✅ Model check passed")
    
    # Infer shapes
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    print("   ✅ Shape inference complete")
    
    # Save inferred model
    onnx.save(onnx_model, str(onnx_inferred))
    
    infer_time = time.time() - start
    print(f"✅ Shape inference complete ({infer_time:.1f}s)")
    
except Exception as e:
    print(f"❌ Shape inference failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# Step 5: Quantize with QDQ format
print("⚡ Step 5: Quantizing to INT8 (QDQ format)...")
print("   This may take 1-2 minutes...")
start = time.time()

try:
    quantize_dynamic(
        str(onnx_inferred),
        str(onnx_int8),
        weight_type=QuantType.QInt8
    )
    
    quant_time = time.time() - start
    print(f"✅ Quantization complete ({quant_time:.1f}s)")
    
    if onnx_int8.exists():
        size_mb = onnx_int8.stat().st_size / (1024 * 1024)
        print(f"   Model file: {size_mb:.1f} MB")
        
        # Check for external data
        external_data = onnx_int8.with_suffix('.onnx.data')
        if external_data.exists():
            ext_size_mb = external_data.stat().st_size / (1024 * 1024)
            print(f"   External data: {ext_size_mb:.1f} MB")
            total_size = size_mb + ext_size_mb
            print(f"   Total: {total_size:.1f} MB")
            
            # Calculate compression
            original_total = onnx_fp32.stat().st_size
            if onnx_fp32.with_suffix('.onnx.data').exists():
                original_total += onnx_fp32.with_suffix('.onnx.data').stat().st_size
            original_mb = original_total / (1024 * 1024)
            compression = (1 - total_size / original_mb) * 100
            print(f"   Compression: {compression:.1f}% smaller")
    else:
        print("❌ Error: Quantized model not created")
        exit(1)
        
except Exception as e:
    print(f"❌ Quantization failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# Step 6: Summary
print("=" * 70)
print("✅ SUCCESS!")
print("=" * 70)
print()
print(f"📁 Quantized model: {onnx_int8}")
print("   Format: ONNX QDQ (mobile-friendly)")
print(f"   Size: ~{total_size:.1f} MB")
print()
print("Next steps:")
print("1. Test: python mobile/scripts/test_onnx_qdq.py")
print("2. Convert to ORT format (optional):")
print("   python -m onnxruntime.tools.convert_onnx_models_to_ort \\")
print(f"       {onnx_int8} --optimize")
print("3. Deploy to Android with ONNX Runtime Mobile")
print()
print("This model should work with onnxruntime-mobile AAR!")
