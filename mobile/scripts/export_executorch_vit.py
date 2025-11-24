"""
ExecuTorch export with PT2E quantization (torch 2.9.x + executorch 1.0.1+cpu).
Bypasses TorchScript/ONNX issues by using torch.export + XNNPACK quantizer.
"""

from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
import time
from pathlib import Path

import torch
from PIL import Image
import numpy as np
from transformers import ViTForImageClassification, ViTImageProcessor

# Use an available quantization backend (torch 2.9.1+cpu exposes only onednn on Windows)
supported = torch.backends.quantized.supported_engines
if not supported:
    raise SystemExit(
        "No quantized backend available in this build of PyTorch.")
torch.backends.quantized.engine = supported[0]

model_dir = Path("models/vit-finetuned")
output_path = Path("mobile/assets/vit_int8_executorch.pte")

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

print("\nStep 3: PT2E quantization with XNNPACK")

quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config())

prepared = prepare_pt2e(training_gm, quantizer)
with torch.no_grad():
    prepared(dummy)  # calibration pass
quantized = convert_pt2e(prepared)
print("PT2E quantization complete")

print("\nStep 4: Lower to XNNPACK delegate and ExecuTorch .pte")

# Export and lower using the official XNNPACK pattern from PyTorch docs
# See: https://pytorch.org/executorch/stable/backends/xnnpack/xnnpack-quantization.html
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
print("\nDone. The .pte uses XNNPACK delegate for optimized INT8 inference on mobile.")
