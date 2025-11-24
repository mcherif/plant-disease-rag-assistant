"""
ExecuTorch export using PT2E quantization and XNNPACK delegate.
Improvements over export_executorch_vit.py:
- Prefer qnnpack backend when available (fallback to fbgemm/first backend).
- Deterministic dummy calibration image (seeded).
- Reuse first torch.export result for quantization, then export quantized module once.
"""

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

import executorch.exir as exir
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

# Set quantization backend with preference for qnnpack on mobile.
engines = list(torch.backends.quantized.supported_engines)
if not engines:
    raise SystemExit("No quantized backend available in this PyTorch build.")
if "qnnpack" in engines:
    torch.backends.quantized.engine = "qnnpack"
elif "fbgemm" in engines:
    torch.backends.quantized.engine = "fbgemm"
else:
    torch.backends.quantized.engine = engines[0]

model_dir = Path("models/vit-finetuned")
output_path = Path("mobile/assets/vit_int8_executorch_qnnpack.pte")

print("=" * 70)
print("ExecuTorch Export with PT2E Quantization (QNNPACK preferred)")
print("=" * 70)
print(f"PyTorch version: {torch.__version__}")
print(f"Quantized backend: {torch.backends.quantized.engine}")

major, minor = map(int, torch.__version__.split(".")[:2])
if (major, minor) < (2, 3):
    raise SystemExit("PyTorch 2.3+ required for torch.export + PT2E")

print("\nStep 1: Load model and deterministic dummy input")
torch.manual_seed(0)
np.random.seed(0)
model = ViTForImageClassification.from_pretrained(model_dir).eval()
processor = ViTImageProcessor.from_pretrained(model_dir)
rand_img = (np.random.rand(224, 224, 3) * 255).astype("uint8")
dummy_image = Image.fromarray(rand_img)
dummy = processor(images=dummy_image, return_tensors="pt")["pixel_values"]
print(f"Dummy input shape: {dummy.shape}")

print("\nStep 2: Export edge graph (torch.export)")
start = time.perf_counter()
edge = torch.export.export(model, (dummy,), {})
export_time = time.perf_counter() - start
print(f"Edge graph exported in {export_time:.1f}s")

print("\nStep 3: PT2E quantization with XNNPACK quantizer")
quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config())

training_gm = edge.module()
start = time.perf_counter()
prepared = prepare_pt2e(training_gm, quantizer)
with torch.no_grad():
    prepared(dummy)  # calibration
quantized = convert_pt2e(prepared)
quant_time = time.perf_counter() - start
print(f"PT2E quantization complete in {quant_time:.1f}s")

print("\nStep 4: Export quantized module and lower to ExecuTorch with XNNPACK")
start = time.perf_counter()
quant_export = torch.export.export(quantized, (dummy,), {})

compile_config = EdgeCompileConfig(delegates=[XnnpackPartitioner()])
et_program = to_edge_transform_and_lower(
    quant_export,
    compile_config=compile_config,
).to_executorch()

lower_time = time.perf_counter() - start

output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "wb") as f:
    et_program.write_to_file(f)

size_mb = output_path.stat().st_size / (1024 * 1024)
print(f"Saved: {output_path} ({size_mb:.1f} MB)")
print(f"Lowering + save time: {lower_time:.1f}s")
print("\nDone. The .pte uses the XNNPACK delegate for mobile INT8 inference.")
