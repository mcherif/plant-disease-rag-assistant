#!/usr/bin/env python3
"""
Quantize ONNX ViT model to INT8 for mobile deployment.

This uses ONNX Runtime's quantization tools to reduce model size
from ~327MB to ~82MB while maintaining accuracy.

Usage:
    python quantize_onnx_model.py --input ../assets/vit_model.onnx --output ../assets/vit_int8.onnx
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def quantize_onnx_model(input_path: Path, output_path: Path):
    """Quantize ONNX model to INT8."""
    logger.info(f"Quantizing ONNX model: {input_path}")
    
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        # Dynamic quantization (weights to INT8)
        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8
        )
        
        # Get file sizes
        input_size_mb = input_path.stat().st_size / (1024 * 1024)
        output_size_mb = output_path.stat().st_size / (1024 * 1024)
        reduction = (1 - output_size_mb / input_size_mb) * 100
        
        logger.info(f"Original size: {input_size_mb:.2f} MB")
        logger.info(f"Quantized size: {output_size_mb:.2f} MB")
        logger.info(f"Size reduction: {reduction:.1f}%")
        logger.info(f"✅ Quantized model saved: {output_path}")
        
    except ImportError:
        logger.error("onnxruntime not installed. Install with: pip install onnxruntime")
        raise


def validate_onnx_model(model_path: Path):
    """Validate ONNX model with sample inference."""
    logger.info("Validating ONNX model...")
    
    try:
        import onnxruntime as ort
        
        # Create inference session
        session = ort.InferenceSession(str(model_path))
        
        # Get input/output info
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_name = session.get_outputs()[0].name
        
        logger.info(f"Input name: {input_name}")
        logger.info(f"Input shape: {input_shape}")
        logger.info(f"Output name: {output_name}")
        
        # Test inference with random input
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs = session.run([output_name], {input_name: dummy_input})
        
        logger.info(f"Output shape: {outputs[0].shape}")
        logger.info(f"Sample output (first 5 logits): {outputs[0][0][:5]}")
        logger.info("✅ ONNX model validation successful")
        
    except ImportError:
        logger.error("onnxruntime not installed. Install with: pip install onnxruntime")
        raise


def main():
    parser = argparse.ArgumentParser(description='Quantize ONNX ViT model to INT8')
    parser.add_argument('--input', type=str, default='../assets/vit_model.onnx',
                        help='Path to input ONNX model')
    parser.add_argument('--output', type=str, default='../assets/vit_int8.onnx',
                        help='Path to output quantized ONNX model')
    
    args = parser.parse_args()
    
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    
    logger.info("=" * 60)
    logger.info("ONNX Model Quantization: FP32 → INT8")
    logger.info("=" * 60)
    
    # Step 1: Quantize
    quantize_onnx_model(input_path, output_path)
    
    # Step 2: Validate
    validate_onnx_model(output_path)
    
    logger.info("=" * 60)
    logger.info(f"✅ Quantization complete: {output_path}")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Use this ONNX model with ONNX Runtime Mobile on Android")
    logger.info("2. Add ONNX Runtime dependency to Android project")
    logger.info("3. Load model and run inference in Kotlin/Java")


if __name__ == '__main__':
    main()
