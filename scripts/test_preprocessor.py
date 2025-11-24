"""
Test script to verify how the ViT preprocessor handles different image resolutions
"""
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

MODEL_DIR = "models/vit-finetuned-15crops-41classes"

# Load model and processor
processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
model.eval()

print("Preprocessor Configuration:")
print(f"  size: {processor.size}")
print(f"  do_resize: {processor.do_resize}")
print(f"  do_normalize: {processor.do_normalize}")
print(f"  image_mean: {processor.image_mean}")
print(f"  image_std: {processor.image_std}")
print()

# Create test images
print("Creating test images...")
# Olive-like image (800x600, bright)
olive_like = Image.new('RGB', (800, 600), color=(180, 180, 180))
# PlantVillage-like image (256x256, darker)
pv_like = Image.new('RGB', (256, 256), color=(120, 120, 120))

# Process both
print("\nProcessing Olive-like image (800x600, mean=180)...")
olive_inputs = processor(images=olive_like, return_tensors="pt")
print(f"  Processed shape: {olive_inputs['pixel_values'].shape}")
print(f"  Processed mean: {olive_inputs['pixel_values'].mean():.4f}")
print(f"  Processed std: {olive_inputs['pixel_values'].std():.4f}")
print(f"  Processed min: {olive_inputs['pixel_values'].min():.4f}")
print(f"  Processed max: {olive_inputs['pixel_values'].max():.4f}")

print("\nProcessing PlantVillage-like image (256x256, mean=120)...")
pv_inputs = processor(images=pv_like, return_tensors="pt")
print(f"  Processed shape: {pv_inputs['pixel_values'].shape}")
print(f"  Processed mean: {pv_inputs['pixel_values'].mean():.4f}")
print(f"  Processed std: {pv_inputs['pixel_values'].std():.4f}")
print(f"  Processed min: {pv_inputs['pixel_values'].min():.4f}")
print(f"  Processed max: {pv_inputs['pixel_values'].max():.4f}")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
print("If both images produce similar processed tensors despite")
print("different input resolutions, then the preprocessor is")
print("working correctly and the issue is elsewhere.")
print("\nIf they produce very different tensors, we have a problem.")
