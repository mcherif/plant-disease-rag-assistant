"""
ExecuTorch export using PT2E INT8 + XNNPACK delegate.
- Prefers qnnpack backend.
- Calibrates on real images (configurable) for better accuracy.
- Reuses torch.export once for quantized module, then lowers with XNNPACK.
"""

import argparse
import sys
import types
from enum import Enum
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.exir import to_edge_transform_and_lower
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from transformers import ViTForImageClassification, ViTImageProcessor

parser = argparse.ArgumentParser()
parser.add_argument(
    "--calib-dir",
    type=str,
    default="data/split/train",
    help="Directory with calibration JPGs.",
)
parser.add_argument(
    "--calib-samples",
    type=int,
    default=512,
    help="Number of calibration images to run observers on.",
)
parser.add_argument(
    "--output",
    type=str,
    default="mobile/assets/vit_int8_executorch_qnnpack.pte",
    help="Output .pte path.",
)
args = parser.parse_args()

# Avoid torchvision custom op registration (e.g., NMS) when using CPU-only builds
# by providing a minimal stub so transformers can import InterpolationMode.


class _InterpolationMode(Enum):
    NEAREST = 0
    BILINEAR = 2
    BICUBIC = 3
    LANCZOS = 1
    HAMMING = 5
    BOX = 4


_tv = types.ModuleType("torchvision")
_tv_transforms = types.ModuleType("torchvision.transforms")
_tv_transforms.InterpolationMode = _InterpolationMode
try:
    import importlib.machinery as _machinery

    _tv.__spec__ = _machinery.ModuleSpec("torchvision", loader=None)
    _tv_transforms.__spec__ = _machinery.ModuleSpec(
        "torchvision.transforms", loader=None
    )
except Exception:
    pass
sys.modules["torchvision"] = _tv
sys.modules["torchvision.transforms"] = _tv_transforms


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
output_path = Path(args.output)

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

# Calibration on real images if available
calib_dir = Path(args.calib_dir)
calib_images = []
if calib_dir.exists():
    calib_images = list(calib_dir.rglob("*.JPG"))[: args.calib_samples]
if calib_images:
    print(f"Calibrating with {len(calib_images)} images from {calib_dir}...")
    with torch.no_grad():
        for p in calib_images:
            img = Image.open(p).convert("RGB")
            t = processor(images=img, return_tensors="pt")["pixel_values"]
            prepared(t)  # per-image to avoid batch shape mismatch
else:
    print("No calibration images found; using dummy for calibration.")
    with torch.no_grad():
        prepared(dummy)

quantized = convert_pt2e(prepared)
quant_time = time.perf_counter() - start
print(f"PT2E quantization complete in {quant_time:.1f}s")

print("\nStep 4: Export quantized module and lower to ExecuTorch with XNNPACK")
start = time.perf_counter()
quant_export = torch.export.export(quantized, (dummy,), {})
et_program = to_edge_transform_and_lower(
    quant_export,
    partitioner=[XnnpackPartitioner()],
).to_executorch()
lower_time = time.perf_counter() - start

output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "wb") as f:
    et_program.write_to_file(f)

size_mb = output_path.stat().st_size / (1024 * 1024)
print(f"Saved: {output_path} ({size_mb:.1f} MB)")
print(f"Lowering + save time: {lower_time:.1f}s")
print("\nDone. The .pte uses the XNNPACK delegate for mobile INT8 inference.")
