#!/usr/bin/env python3
"""
Simplified ViT model conversion: PyTorch → ONNX → TFLite (INT8)

This version skips the problematic onnx-tf step and uses a direct approach.

Usage:
    python convert_vit_simple.py --input ../../models/vit-finetuned --output ../assets/vit_int8.tflite
"""

import argparse
import logging
from pathlib import Path

import torch
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


def convert_to_torchscript(model, output_path: Path):
    """Convert PyTorch model to TorchScript for mobile."""
    logger.info(f"Converting to TorchScript: {output_path}")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Optimize for mobile
    from torch.utils.mobile_optimizer import optimize_for_mobile
    optimized_model = optimize_for_mobile(traced_model)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    optimized_model._save_for_lite_interpreter(str(output_path))
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"TorchScript model saved: {output_path} ({size_mb:.2f} MB)")
    
    return optimized_model


def validate_torchscript(model_path: Path):
    """Validate TorchScript model."""
    logger.info("Validating TorchScript model...")
    
    # Load model
    model = torch.jit.load(str(model_path))
    model.eval()
    
    # Test inference
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Sample output (first 5 logits): {output[0][:5]}")
    logger.info("✅ TorchScript model validation successful")


def main():
    parser = argparse.ArgumentParser(description='Convert ViT model to TorchScript (mobile)')
    parser.add_argument('--input', type=str, default='../../models/vit-finetuned',
                        help='Path to PyTorch model directory')
    parser.add_argument('--output', type=str, default='../assets/vit_mobile.ptl',
                        help='Path to output TorchScript model')
    
    args = parser.parse_args()
    
    model_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    
    logger.info("=" * 60)
    logger.info("ViT Model Conversion: PyTorch → TorchScript (Mobile)")
    logger.info("=" * 60)
    
    # Step 1: Load PyTorch model
    model, processor = load_pytorch_model(model_path)
    
    # Step 2: Convert to TorchScript
    convert_to_torchscript(model, output_path)
    
    # Step 3: Validate
    validate_torchscript(output_path)
    
    logger.info("=" * 60)
    logger.info(f"✅ Conversion complete: {output_path}")
    logger.info("=" * 60)
    logger.info("\nNote: This creates a PyTorch Mobile model (.ptl)")
    logger.info("For TFLite, we need a different approach due to onnx-tf compatibility issues.")
    logger.info("Consider using this PyTorch Mobile model for Android instead.")


if __name__ == '__main__':
    main()
