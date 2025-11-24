"""
Best-effort runner for vit_int8_executorch_qnnpack.pte over an image directory.
- Tries ExecuTorch ExecutionSession; if unavailable, falls back to FP32 ViT.
- Writes predictions to mobile/assets/predictions_qnnpack.csv.
"""

from pathlib import Path
import csv
import sys

import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor


MODEL_DIR = Path("models/vit-finetuned")
PTE_PATH = Path("mobile/assets/vit_int8_executorch_qnnpack.pte")
IMAGE_ROOT = Path("data/split")
OUTPUT_CSV = Path("mobile/assets/predictions_qnnpack.csv")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            yield p


def try_executorch_session(pte_path: Path):
    try:
        try:
            from executorch.runtime import ExecutionSession  # type: ignore
        except Exception:
            from executorch.runtime.runtime import ExecutionSession  # type: ignore
        return ExecutionSession(str(pte_path))
    except Exception:
        return None


def load_processor_and_model():
    processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
    model = ViTForImageClassification.from_pretrained(MODEL_DIR).eval()
    return processor, model


def main():
    if not IMAGE_ROOT.exists():
        print(f"Image root not found: {IMAGE_ROOT}")
        sys.exit(1)

    processor, model = load_processor_and_model()
    id2label = model.config.id2label

    session = None
    runtime_mode = "executorch"
    if PTE_PATH.exists():
        session = try_executorch_session(PTE_PATH)
    if session is None:
        print("=== ExecuTorch runtime not available; FALLING BACK TO FP32 ViT ===")
        runtime_mode = "torch-fp32"

    rows = [("image", "label", "score", "mode")]
    for img_path in iter_images(IMAGE_ROOT):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:  # noqa: BLE001
            print(f"Skip {img_path}: {e}")
            continue

        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"]

        if session is not None:
            try:
                out = session.run_method("forward", pixel_values)[0]
                logits = torch.tensor(out)  # convert ExecuTorch tensor to torch tensor
            except Exception as e:  # noqa: BLE001
                print(f"ExecuTorch failed on {img_path}: {e}; fallback to torch.")
                with torch.no_grad():
                    logits = model(**inputs).logits
        else:
            with torch.no_grad():
                logits = model(**inputs).logits

        scores = torch.softmax(logits, dim=-1)
        score, pred = torch.max(scores, dim=-1)
        label = id2label.get(pred.item(), str(pred.item()))
        rows.append((str(img_path), label, float(score.item()), runtime_mode))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Wrote {OUTPUT_CSV} with {len(rows)-1} predictions (mode={runtime_mode}).")


if __name__ == "__main__":
    main()
