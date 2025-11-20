import shutil
import random
from pathlib import Path

# Setup paths
train_dir = Path("data/split/train")
val_dir = Path("data/split/val")
val_ratio = 0.2
random.seed(42)

# Olive classes to process
olive_classes = ["Olive___Aculus_olearius", "Olive___Peacock_spot", "Olive___healthy"]

print("Fixing Olive data split...")

for class_name in olive_classes:
    src_dir = train_dir / class_name
    dst_dir = val_dir / class_name
    
    # Ensure destination exists
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images in train (handle case-insensitivity)
    images = set(list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.JPG")))
    images = list(images)
    count = len(images)
    
    if count == 0:
        print(f"Warning: No images found in {src_dir}")
        continue
        
    # Calculate split
    num_val = int(count * val_ratio)
    val_images = random.sample(images, num_val)
    
    print(f"{class_name}: Moving {num_val} of {count} images to validation...")
    
    # Move files
    for img in val_images:
        shutil.move(str(img), str(dst_dir / img.name))

print("\nOlive data split complete!")
