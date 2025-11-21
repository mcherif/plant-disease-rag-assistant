#!/usr/bin/env python3
"""
Convert ViT model from PyTorch/Safetensors to TensorFlow Lite with INT8 quantization.

Conversion path: PyTorch (.safetensors) → ONNX → TensorFlow → TFLite (INT8)

Usage:
    python convert_vit_to_tflite.py --input ../../models/vit-finetuned --output ../assets/vit_int8.tflite
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import onnx
import tensorflow as tf
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_pytorch_model(model_path: Path):
    """Load the PyTorch ViT model from safetensors."""
    logger.info(f"Loading PyTorch model from {model_path}")
    model = ViTForImageClassification.from_pretrained(str(model_path))
    processor = ViTImageProcessor.from_pretrained(str(model_path))
    model.eval()
    return model, processor


def export_to_onnx(model, processor, onnx_path: Path):
    """Export PyTorch model to ONNX format."""
    logger.info(f"Exporting to ONNX: {onnx_path}")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['logits'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verified successfully")


def onnx_to_tensorflow(onnx_path: Path, tf_saved_model_path: Path):
    """Convert ONNX model to TensorFlow SavedModel."""
    logger.info(f"Converting ONNX to TensorFlow SavedModel: {tf_saved_model_path}")
    
    try:
        import onnx_tf
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = onnx_tf.backend.prepare(onnx_model)
        tf_rep.export_graph(str(tf_saved_model_path))
        logger.info("TensorFlow SavedModel created successfully")
    except ImportError:
        logger.error("onnx-tf not installed. Install with: pip install onnx-tf")
        raise


def create_representative_dataset(model_path: Path, num_samples: int = 100):
    """Create representative dataset for INT8 calibration."""
    logger.info(f"Creating representative dataset with {num_samples} samples")
    
    processor = ViTImageProcessor.from_pretrained(str(model_path))
    
    # Generate random images (in production, use real validation images)
    def representative_data_gen():
        for _ in range(num_samples):
            # Random RGB image
            random_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            pil_image = Image.fromarray(random_image)
            
            # Preprocess
            inputs = processor(images=pil_image, return_tensors="np")
            pixel_values = inputs['pixel_values'].astype(np.float32)
            
            yield [pixel_values]
    
    return representative_data_gen


def convert_to_tflite_int8(tf_saved_model_path: Path, tflite_path: Path, model_path: Path):
    """Convert TensorFlow SavedModel to TFLite with INT8 quantization."""
    logger.info(f"Converting to TFLite with INT8 quantization: {tflite_path}")
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_saved_model_path))
    
    # Enable INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = create_representative_dataset(model_path)
    
    # Ensure full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = len(tflite_model) / (1024 * 1024)
    logger.info(f"TFLite model saved: {tflite_path} ({size_mb:.2f} MB)")


def validate_tflite_model(tflite_path: Path, model_path: Path):
    """Validate TFLite model with sample inference."""
    logger.info("Validating TFLite model...")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    logger.info(f"Input shape: {input_details[0]['shape']}")
    logger.info(f"Input dtype: {input_details[0]['dtype']}")
    logger.info(f"Output shape: {output_details[0]['shape']}")
    logger.info(f"Output dtype: {output_details[0]['dtype']}")
    
    # Test inference with random input
    input_shape = input_details[0]['shape']
    input_data = np.random.randint(0, 256, input_shape, dtype=np.uint8)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    logger.info(f"Output shape: {output_data.shape}")
    logger.info(f"Sample output (first 5 logits): {output_data[0][:5]}")
    
    logger.info("✅ TFLite model validation successful")


def main():
    parser = argparse.ArgumentParser(description='Convert ViT model to TFLite INT8')
    parser.add_argument('--input', type=str, default='../../models/vit-finetuned',
                        help='Path to PyTorch model directory')
    parser.add_argument('--output', type=str, default='../assets/vit_int8.tflite',
                        help='Path to output TFLite model')
    parser.add_argument('--keep-intermediate', action='store_true',
                        help='Keep intermediate ONNX and TF SavedModel files')
    
    args = parser.parse_args()
    
    model_path = Path(args.input).resolve()
    tflite_path = Path(args.output).resolve()
    
    # Intermediate paths
    onnx_path = tflite_path.parent / 'vit_model.onnx'
    tf_saved_model_path = tflite_path.parent / 'vit_saved_model'
    
    logger.info("=" * 60)
    logger.info("ViT Model Conversion: PyTorch → ONNX → TensorFlow → TFLite")
    logger.info("=" * 60)
    
    # Step 1: Load PyTorch model
    model, processor = load_pytorch_model(model_path)
    
    # Step 2: Export to ONNX
    export_to_onnx(model, processor, onnx_path)
    
    # Step 3: Convert ONNX to TensorFlow
    onnx_to_tensorflow(onnx_path, tf_saved_model_path)
    
    # Step 4: Convert to TFLite with INT8 quantization
    convert_to_tflite_int8(tf_saved_model_path, tflite_path, model_path)
    
    # Step 5: Validate
    validate_tflite_model(tflite_path, model_path)
    
    # Cleanup intermediate files
    if not args.keep_intermediate:
        logger.info("Cleaning up intermediate files...")
        if onnx_path.exists():
            onnx_path.unlink()
        if tf_saved_model_path.exists():
            import shutil
            shutil.rmtree(tf_saved_model_path)
    
    logger.info("=" * 60)
    logger.info(f"✅ Conversion complete: {tflite_path}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
