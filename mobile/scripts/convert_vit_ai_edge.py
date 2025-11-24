import torch
from ai_edge_torch import export_to_tflite
from pathlib import Path
from PIL import Image
import numpy as np
from transformers import ViTForImageClassification

# Load the HuggingFace ViT model from SafeTensors
model_path = Path("../../models/vit-finetuned")  # relative to script location
print(f"Loading model from {model_path.absolute()}...")
model = ViTForImageClassification.from_pretrained(str(model_path))
model.eval()
print("✅ Model loaded successfully")

# Simple preprocessing using PIL and numpy (mirrors torchvision transforms)
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    # Normalize using ImageNet means/std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    # Convert to (C, H, W)
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr).unsqueeze(0)


# Representative dataset for INT8 calibration using the custom preprocess_image
def representative_dataset():
    img_dir = Path("../../mobile/data/calibration_images")
    # Get all jpg and JPG files
    jpg_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.JPG"))
    print(f"Found {len(jpg_files)} calibration images")
    for img_path in jpg_files[:10]:
        print(f"  Processing {img_path.name}...")
        tensor = preprocess_image(img_path)
        yield {"pixel_values": tensor}

# Export to TFLite with INT8 quantization
output_path = Path("../../mobile/assets/vit_model_ai_edge.tflite")
print("\n🚀 Starting INT8 quantization and TFLite export...")
print(f"   Output path: {output_path.absolute()}")
export_to_tflite(
    model=model,
    input_spec={"pixel_values": (1, 3, 224, 224)},
    output_path=str(output_path),
    quantization="int8",
    calibration_dataset=representative_dataset(),
)
print(f"\n✅ TFLite model saved to {output_path.absolute()}")
