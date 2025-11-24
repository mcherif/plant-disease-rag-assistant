"""
Validate PT2E quantized model against original FP32 model.
Tests on a single image to compare outputs before ExecuTorch export.
"""

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

MODEL_DIR = Path("models/vit-finetuned")
TEST_IMAGE = Path("data/split/val")  # Will pick first image found


def find_test_image(root: Path) -> Path:
    """Find first valid image in directory."""
    for ext in [".jpg", ".jpeg", ".png"]:
        for img in root.rglob(f"*{ext}"):
            if img.is_file():
                return img
    raise FileNotFoundError(f"No test images found in {root}")


def main():
    print("=" * 70)
    print("PT2E Quantization Validation")
    print("=" * 70)
    
    # Set deterministic seeds
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Load models
    print("\n1. Loading original FP32 model...")
    model_fp32 = ViTForImageClassification.from_pretrained(MODEL_DIR).eval()
    processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
    id2label = model_fp32.config.id2label
    
    # Create quantized model
    print("2. Creating PT2E quantized model...")
    start = time.time()
    
    # Prepare dummy input for export
    rand_img = (np.random.rand(224, 224, 3) * 255).astype("uint8")
    dummy = processor(images=Image.fromarray(rand_img), return_tensors="pt")["pixel_values"]
    
    # Export and quantize
    edge = torch.export.export(model_fp32, (dummy,), {})
    training_gm = edge.module()
    
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config())
    
    prepared = prepare_pt2e(training_gm, quantizer)
    with torch.no_grad():
        prepared(dummy)  # calibration
    model_int8 = convert_pt2e(prepared)
    
    print(f"   Quantization completed in {time.time() - start:.1f}s")
    
    # Load test image
    print("\n3. Loading test image...")
    test_img_path = find_test_image(TEST_IMAGE)
    print(f"   Using: {test_img_path}")
    
    test_image = Image.open(test_img_path).convert("RGB")
    inputs = processor(images=test_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    
    # Run inference on both models
    print("\n4. Running inference...")
    
    with torch.no_grad():
        # FP32 model
        start = time.time()
        outputs_fp32 = model_fp32(**inputs)
        time_fp32 = time.time() - start
        logits_fp32 = outputs_fp32.logits
        
        # INT8 model
        start = time.time()
        logits_int8 = model_int8(pixel_values)
        time_int8 = time.time() - start
    
    # Compare results
    print("\n5. Comparison Results:")
    print("-" * 70)
    
    # Get predictions
    probs_fp32 = torch.softmax(logits_fp32, dim=-1)
    probs_int8 = torch.softmax(logits_int8, dim=-1)
    
    pred_fp32 = torch.argmax(probs_fp32, dim=-1).item()
    pred_int8 = torch.argmax(probs_int8, dim=-1).item()
    
    conf_fp32 = probs_fp32[0, pred_fp32].item()
    conf_int8 = probs_int8[0, pred_int8].item()
    
    label_fp32 = id2label[pred_fp32]
    label_int8 = id2label[pred_int8]
    
    print("FP32 Model:")
    print(f"  Prediction: {label_fp32}")
    print(f"  Confidence: {conf_fp32:.4f}")
    print(f"  Inference time: {time_fp32*1000:.2f}ms")
    
    print("\nINT8 Model (PT2E):")
    print(f"  Prediction: {label_int8}")
    print(f"  Confidence: {conf_int8:.4f}")
    print(f"  Inference time: {time_int8*1000:.2f}ms")
    
    # Check if predictions match
    match = "✓ MATCH" if pred_fp32 == pred_int8 else "✗ MISMATCH"
    print(f"\nPrediction Match: {match}")
    
    # Calculate logits difference (SQNR-like metric)
    logits_diff = torch.abs(logits_fp32 - logits_int8).mean().item()
    logits_max_diff = torch.abs(logits_fp32 - logits_int8).max().item()
    
    print("\nLogits Difference:")
    print(f"  Mean absolute diff: {logits_diff:.6f}")
    print(f"  Max absolute diff: {logits_max_diff:.6f}")
    
    # Top-5 comparison
    print("\nTop-5 Predictions:")
    top5_fp32 = torch.topk(probs_fp32, 5, dim=-1)
    top5_int8 = torch.topk(probs_int8, 5, dim=-1)
    
    print(f"  FP32: {[id2label[i.item()] for i in top5_fp32.indices[0]]}")
    print(f"  INT8: {[id2label[i.item()] for i in top5_int8.indices[0]]}")
    
    print("\n" + "=" * 70)
    print("Validation Complete!")
    print("=" * 70)
    
    if pred_fp32 == pred_int8:
        print("✓ Quantization preserved prediction on this test image.")
        print("  The ExecuTorch .pte file should work correctly on Android.")
    else:
        print("⚠ Quantization changed the prediction on this test image.")
        print("  Consider testing on more images or adjusting quantization config.")


if __name__ == "__main__":
    main()
