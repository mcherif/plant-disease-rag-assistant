"""
ExecuTorch export utility for ViT.
Supports both FP32 (no quantization) and PT2E INT8 export using XNNPACK.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-quantize",
    action="store_true",
    help="Export FP32 (no quantization). Default is INT8 PT2E quantization.",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Optional output path; defaults to mobile/assets/vit_int8_executorch.pte or vit_fp32_executorch.pte",
)
parser.add_argument(
    "--calib-dir",
    type=str,
    default="data/split/train",
    help="Directory with calibration images (JPG). Used only for INT8.",
)
parser.add_argument(
    "--calib-samples",
    type=int,
    default=16,
    help="Number of calibration images to run observers on (INT8 only).",
)
args = parser.parse_args()

# Use an available quantization backend (torch 2.9.x+cpu exposes only onednn on Windows)
if not args.no_quantize:
    supported = torch.backends.quantized.supported_engines
    if not supported:
        raise SystemExit(
            "No quantized backend available in this build of PyTorch.")
    torch.backends.quantized.engine = supported[0]

model_dir = Path("models/vit-finetuned")
default_name = "vit_fp32_executorch.pte" if args.no_quantize else "vit_int8_executorch.pte"
output_path = Path(args.output) if args.output else Path("mobile/assets") / default_name

print("=" * 70)
print("ExecuTorch Export with PT2E Quantization")
print("=" * 70)
print(f"PyTorch version: {torch.__version__}")

major, minor = map(int, torch.__version__.split(".")[:2])
if (major, minor) < (2, 3):
    raise SystemExit("PyTorch 2.3+ required for torch.export + PT2E")

print("\nStep 1: Load model and dummy input")
model = ViTForImageClassification.from_pretrained(model_dir).eval()
processor = ViTImageProcessor.from_pretrained(model_dir)
# Create a safe dummy image in uint8 range to avoid preprocessing errors
rand_img = (np.random.rand(224, 224, 3) * 255).astype("uint8")
dummy_image = Image.fromarray(rand_img)
dummy = processor(images=dummy_image, return_tensors="pt")["pixel_values"]
print(f"Dummy input shape: {dummy.shape}")

print("\nStep 2: Export edge graph (torch.export)")
start = time.time()
edge = torch.export.export(model, (dummy,), {})
training_gm = edge.module()  # Get GraphModule for PT2E quantization
print(f"Edge graph exported in {time.time() - start:.1f}s")

if args.no_quantize:
    print("\nStep 3: Skip quantization (FP32 export)")
    quantized = training_gm  # just reuse the FP32 module
else:
    print("\nStep 3: PT2E quantization with XNNPACK")
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config())
    prepared = prepare_pt2e(training_gm, quantizer)
    # Calibration on real images (if available)
    calib_dir = Path(args.calib_dir)
    calib_images = []
    if calib_dir.exists():
        calib_images = list(calib_dir.rglob("*.JPG"))[: args.calib_samples]
    if calib_images:
        print(f"Calibrating with {len(calib_images)} images from {calib_dir} (batch size 1)...")
        with torch.no_grad():
            for p in calib_images:
                img = Image.open(p).convert("RGB")
                t = processor(images=img, return_tensors="pt")["pixel_values"]
                prepared(t)  # run observers per-image to avoid static batch mismatch
    else:
        print("No calibration images found; using dummy for calibration.")
        with torch.no_grad():
            prepared(dummy)  # calibration pass
    quantized = convert_pt2e(prepared)
    print("PT2E quantization complete")

print("\nStep 4: Lower to XNNPACK delegate and ExecuTorch .pte")

# Always lower with XNNPACK so FP32 also benefits from optimized kernels.
et_program = to_edge_transform_and_lower(
    torch.export.export(quantized, (dummy,)),
    partitioner=[XnnpackPartitioner()],
).to_executorch()

# Save the ExecuTorch program
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "wb") as file:
    et_program.write_to_file(file)

size_mb = output_path.stat().st_size / (1024 * 1024)

print(f"Saved: {output_path} ({size_mb:.1f} MB)")
if args.no_quantize:
    print("\nDone. Exported FP32 ExecuTorch program (no quantization).")
else:
    print("\nDone. Exported INT8 ExecuTorch program with XNNPACK delegate.")
