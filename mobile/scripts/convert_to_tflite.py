"""
Convert ONNX model to TensorFlow Lite with INT8 quantization
Pipeline: ONNX → TensorFlow SavedModel → TFLite INT8
"""

import tensorflow as tf
from pathlib import Path
import onnx
from onnx_tf.backend import prepare
import numpy as np
from PIL import Image
import time

print("=" * 70)
print("TensorFlow Lite INT8 Conversion")
print("=" * 70)
print()

# Paths
onnx_path = Path("mobile/assets/vit_model_fixed.onnx")
tf_model_dir = Path("mobile/assets/vit_tf_model")
tflite_path = Path("mobile/assets/vit_model_int8.tflite")
calib_dir = Path("data/split/val")

# Step 1: Load ONNX model
print("📦 Step 1: Loading ONNX model...")
if not onnx_path.exists():
    print(f"❌ Error: ONNX model not found at {onnx_path}")
    print("   Please run: python mobile/scripts/export_onnx_fixed.py first")
    exit(1)

start = time.time()
onnx_model = onnx.load(str(onnx_path))
print(f"✅ ONNX model loaded ({time.time() - start:.2f}s)")
print()

# Step 2: Convert ONNX → TensorFlow
print("🔄 Step 2: Converting ONNX → TensorFlow...")
start = time.time()
try:
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(str(tf_model_dir))
    print(f"✅ TensorFlow model exported ({time.time() - start:.2f}s)")
except Exception as e:
    print(f"❌ ONNX → TF conversion failed: {e}")
    print("   This may be due to unsupported ONNX operators")
    exit(1)
print()

# Step 3: Create representative dataset for calibration
print("📊 Step 3: Creating calibration dataset...")

def representative_dataset():
    """Generate calibration data for INT8 quantization"""
    # Collect calibration images
    calib_images = []
    for class_dir in calib_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.JPG"))[:2]  # 2 per class
            calib_images.extend(images)
    
    calib_images = calib_images[:50]  # Use 50 images total
    print(f"   Using {len(calib_images)} calibration images")
    
    for i, img_path in enumerate(calib_images):
        if (i + 1) % 10 == 0:
            print(f"   Processing {i+1}/{len(calib_images)}...", end='\r')
        
        # Load and preprocess image
        image = Image.open(img_path).convert("RGB")
        image = image.resize((224, 224))
        
        # Convert to array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        img_array = img_array.transpose(2, 0, 1)  # HWC → CHW
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dim
        
        yield [img_array.astype(np.float32)]
    
    print("\n✅ Calibration dataset ready")

print()

# Step 4: Convert TensorFlow → TFLite with INT8
print("🚀 Step 4: Converting to TFLite with INT8 quantization...")
start = time.time()

try:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_dir))
    
    # Enable INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    convert_time = time.time() - start
    
    if tflite_path.exists():
        size_mb = tflite_path.stat().st_size / (1024 * 1024)
        print()
        print("=" * 70)
        print("✅ TFLITE CONVERSION SUCCESSFUL!")
        print("=" * 70)
        print(f"📁 Output: {tflite_path}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Conversion time: {convert_time:.1f}s")
        print()
        print("Next steps:")
        print("1. Test: python mobile/scripts/test_tflite.py")
        print("2. Deploy to Android app")
    else:
        print("❌ Error: TFLite model not created")
        exit(1)
        
except Exception as e:
    print(f"❌ TFLite conversion failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
