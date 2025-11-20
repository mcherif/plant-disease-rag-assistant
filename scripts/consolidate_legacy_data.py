import shutil
import json
from pathlib import Path
from datetime import datetime

source = Path("C:/projects/PlantVillage-Dataset/raw/color")
dest_train = Path("data/split/train")
metadata_file = Path("data/split/data_sources.json")

# Initialize or load metadata
if metadata_file.exists():
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
else:
    metadata = {"sources": {}, "last_updated": None}

# Move all legacy crop folders to train
moved_count = 0
for crop_dir in source.iterdir():
    if crop_dir.is_dir():
        dest = dest_train / crop_dir.name
        if not dest.exists():
            shutil.move(str(crop_dir), str(dest))
            
            # Track data source
            metadata["sources"][crop_dir.name] = {
                "origin": "PlantVillage",
                "original_path": str(crop_dir),
                "moved_date": datetime.now().isoformat(),
                "image_count": len(list(dest.glob("*.jpg"))) + len(list(dest.glob("*.JPG")))
            }
            moved_count += 1
            print(f"Moved {crop_dir.name}")

# Add Olive metadata if not already present
for olive_class in ["Olive___healthy", "Olive___Aculus_olearius", "Olive___Peacock_spot"]:
    if olive_class not in metadata["sources"]:
        olive_dir = dest_train / olive_class
        if olive_dir.exists():
            metadata["sources"][olive_class] = {
                "origin": "Kaggle (sinanuguz/CNN_olive_dataset + habibulbasher01644/olive-leaf-image-dataset)",
                "original_path": "data/raw/olive_dataset/archive/dataset",
                "added_date": datetime.now().isoformat(),
                "image_count": len(list(olive_dir.glob("*.jpg"))) + len(list(olive_dir.glob("*.JPG")))
            }

metadata["last_updated"] = datetime.now().isoformat()
metadata["total_classes"] = len(metadata["sources"])

# Save metadata
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nMoved {moved_count} classes from PlantVillage")
print(f"Total classes: {metadata['total_classes']}")
print(f"Metadata saved to: {metadata_file}")
