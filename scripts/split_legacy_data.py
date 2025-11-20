import random
import shutil
from pathlib import Path

train_dir = Path("data/split/train")
val_dir = Path("data/split/val")
val_ratio = 0.2
random.seed(42)  # For reproducibility

# Process only non-Olive classes (legacy PlantVillage data)
for class_dir in train_dir.iterdir():
    if class_dir.is_dir() and not class_dir.name.startswith("Olive"):
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG"))
        random.shuffle(images)
        
        split_idx = int(len(images) * (1 - val_ratio))
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Move validation images
        val_class_dir = val_dir / class_dir.name
        val_class_dir.mkdir(exist_ok=True)
        
        for img in val_images:
            dest_path = val_class_dir / img.name
            try:
                shutil.move(str(img), str(dest_path))
            except Exception as e:
                print(f"Error moving {img.name}: {e}")
                continue
        
        print(f"{class_dir.name}: {len(train_images)} train, {len(val_images)} val")

print("\nSplit complete!")
