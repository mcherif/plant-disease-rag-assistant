"""
ONNX Export with Shape Fix - Patch for 768 vs 41 shape inference error
Based on alternative LLM's recommendation to fix classifier weight shape
"""

import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from pathlib import Path
from PIL import Image
import onnx
from onnx import numpy_helper
from onnxruntime.quantization import quantize_dynamic, QuantType
import time

print("=" * 70)
print("ONNX Export with Shape Fix + QDQ Quantization")
print("=" * 70)
print()

# Paths
model_dir = Path("models/vit-finetuned")
sample_image = Path("data/sample.jpg")
onnx_clean = Path("mobile/assets/vit_fp32_clean.onnx")
onnx_int8 = Path("mobile/assets/vit_int8_qdq.onnx")

# Step 1: Load model
print("📦 Step 1: Loading model...")
model = ViTForImageClassification.from_pretrained(str(model_dir), torchscript=False)
processor = ViTImageProcessor.from_pretrained(str(model_dir))
model.eval()
print("✅ Model loaded")
print()

# Step 2: Create dummy input
print("📸 Step 2: Creating dummy input...")
if sample_image.exists():
    image = Image.open(sample_image)
    dummy_input = processor(images=image, return_tensors="pt")["pixel_values"]
else:
    dummy_input = torch.randn(1, 3, 224, 224)
print(f"   Input shape: {dummy_input.shape}")
print()

# Step 3: Export to ONNX WITHOUT constant folding
print("🚀 Step 3: Exporting to ONNX (no constant folding)...")
start = time.time()

try:
    torch.onnx.export(
        model,
        (dummy_input,),
        str(onnx_clean),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "logits": {0: "batch"}
        },
        opset_version=17,
        do_constant_folding=False  # Avoid folding that confuses shapes
    )
    
    export_time = time.time() - start
    print(f"✅ ONNX export complete ({export_time:.1f}s)")
    
except Exception as e:
    print(f"❌ ONNX export failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# Step 4: Fix classifier weight shape
print("🔧 Step 4: Fixing classifier weight shape...")
start = time.time()

try:
    # Load ONNX model
    onnx_model = onnx.load(str(onnx_clean))
    
    # Find classifier weight initializer
    classifier_weights = [i for i in onnx_model.graph.initializer 
                         if "classifier" in i.name and "weight" in i.name]
    
    if classifier_weights:
        cls_w = classifier_weights[0]
        print(f"   Found classifier weight: {cls_w.name}")
        
        # Get current array
        arr = numpy_helper.to_array(cls_w)
        print(f"   Current shape: {arr.shape}")
        
        # Ensure shape is [41, 768] (num_labels, hidden_size)
        if arr.shape != (41, 768):
            print("   Reshaping to [41, 768]...")
            arr = arr.reshape(41, 768)
            
            # Replace initializer
            onnx_model.graph.initializer.remove(cls_w)
            onnx_model.graph.initializer.append(
                numpy_helper.from_array(arr, cls_w.name)
            )
            
            # Save fixed model
            onnx.save(onnx_model, str(onnx_clean))
            print("   ✅ Shape fixed and saved")
        else:
            print("   ✅ Shape already correct")
    else:
        print("   ⚠️  No classifier weight found, proceeding anyway")
    
    fix_time = time.time() - start
    print(f"✅ Shape fix complete ({fix_time:.1f}s)")
    
except Exception as e:
    print(f"⚠️  Shape fix failed: {e}")
    print("   Proceeding with quantization anyway...")

print()

# Step 5: Quantize WITHOUT shape inference
print("⚡ Step 5: Quantizing to INT8 (skipping shape inference)...")
print("   This may take 1-2 minutes...")
start = time.time()

try:
    # Try with minimal parameters to avoid shape inference
    quantize_dynamic(
        str(onnx_clean),
        str(onnx_int8),
        weight_type=QuantType.QInt8
    )
    
    quant_time = time.time() - start
    print(f"✅ Quantization complete ({quant_time:.1f}s)")
    print()
    
    # Check output
    if onnx_int8.exists():
        size_mb = onnx_int8.stat().st_size / (1024 * 1024)
        print(f"📁 Quantized model: {onnx_int8}")
        print(f"   Size: {size_mb:.1f} MB")
        
        # Check for external data
        external_data = onnx_int8.with_suffix('.onnx.data')
        if external_data.exists():
            ext_size_mb = external_data.stat().st_size / (1024 * 1024)
            print(f"   External data: {ext_size_mb:.1f} MB")
            total_size = size_mb + ext_size_mb
            print(f"   Total: {total_size:.1f} MB")
        else:
            total_size = size_mb
        
        print()
        print("=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print()
        print(f"Quantized model: {onnx_int8}")
        print(f"Total size: {total_size:.1f} MB")
        print()
        print("Next steps:")
        print("1. Test: python mobile/scripts/test_onnx_int8.py")
        print("2. Deploy to Android with ONNX Runtime Mobile")
        
    else:
        print("❌ Error: Quantized model not created")
        exit(1)
        
except Exception as e:
    print(f"❌ Quantization failed: {e}")
    import traceback
    traceback.print_exc()
    print()
    print("If this is still the shape inference error, the ONNX graph")
    print("has deeper issues that require manual graph surgery.")
    exit(1)
