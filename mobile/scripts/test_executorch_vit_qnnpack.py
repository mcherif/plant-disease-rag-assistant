"""
Best-effort sanity check for vit_int8_executorch_qnnpack.pte.
- Verifies the file exists/size.
- Attempts to load with ExecuTorch Python runtime if available and run a dummy forward.
  (Many pip builds omit the Python ExecutionSession; in that case we just report it.)
"""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import ViTImageProcessor

MODEL_PATH = Path("mobile/assets/vit_int8_executorch_qnnpack.pte")
MODEL_DIR = Path("models/vit-finetuned")


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing model file: {MODEL_PATH}")

    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"Found {MODEL_PATH} ({size_mb:.1f} MB)")

    # Prepare deterministic dummy input matching export preprocessing.
    torch.manual_seed(0)
    np.random.seed(0)
    processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
    rand_img = (np.random.rand(224, 224, 3) * 255).astype("uint8")
    dummy = processor(images=Image.fromarray(rand_img), return_tensors="pt")["pixel_values"]

    # Try to load via ExecuTorch runtime if available.
    session = None
    err = None
    try:
        # Common entry points in ExecuTorch releases; guarded for missing runtime.
        try:
            from executorch.runtime import ExecutionSession  # type: ignore
        except Exception:
            from executorch.runtime.runtime import ExecutionSession  # type: ignore
        session = ExecutionSession(str(MODEL_PATH))
    except Exception as e:  # noqa: BLE001
        err = e

    if session is None:
        print("ExecuTorch Python runtime not available; skipping runtime test.")
        if err:
            print(f"Runtime import error: {err}")
        return

    try:
        out = session.run_method("forward", dummy)[0]
        print("Runtime forward() ok; output shape:", getattr(out, "shape", type(out)))
    except Exception as e:  # noqa: BLE001
        print(f"Runtime test failed: {e}")


if __name__ == "__main__":
    main()
