import shutil
from pathlib import Path

def organize_dataset():
    base_dir = Path("data/raw/olive_dataset")
    archive_dir = base_dir / "archive/dataset"
    split_dir = Path("data/split")
    
    # Define mappings
    # Source -> (Destination Split, Destination Class)
    moves = [
        # Original dataset paths
        (base_dir / "Healthy", "train", "Olive___healthy"),
        (base_dir / "aculus_olearius", "train", "Olive___Aculus_olearius"),
        (base_dir / "train3", "train", "Olive___Aculus_olearius"),
        (base_dir / "train4", "train", "Olive___Peacock_spot"),
        (base_dir / "test/Healthy", "val", "Olive___healthy"),
        (base_dir / "test/aculus_olearius", "val", "Olive___Aculus_olearius"),
        (base_dir / "test/olive_peacock_spot", "val", "Olive___Peacock_spot"),
        
        # New user-provided dataset paths
        (archive_dir / "train/Healthy", "train", "Olive___healthy"),
        (archive_dir / "train/aculus_olearius", "train", "Olive___Aculus_olearius"),
        (archive_dir / "train/olive_peacock_spot", "train", "Olive___Peacock_spot"),
        (archive_dir / "test/Healthy", "val", "Olive___healthy"),
        (archive_dir / "test/aculus_olearius", "val", "Olive___Aculus_olearius"),
        (archive_dir / "test/olive_peacock_spot", "val", "Olive___Peacock_spot"),
    ]

    for src, split, class_name in moves:
        if not src.exists():
            print(f"Warning: Source {src} does not exist. Skipping.")
            continue
            
        dest_dir = split_dir / split / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {src} -> {dest_dir}...")
        
        # Move all files from source to destination
        count = 0
        for item in src.iterdir():
            if item.is_file():
                # Handle potential duplicates by renaming
                dest_path = dest_dir / item.name
                if dest_path.exists():
                    # Append a suffix if file exists
                    stem = item.stem
                    suffix = item.suffix
                    counter = 1
                    while dest_path.exists():
                        dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                
                try:
                    shutil.move(str(item), str(dest_path))
                    count += 1
                except Exception as e:
                    print(f"Error moving {item}: {e}")
        
        print(f"Moved {count} files from {src}")

if __name__ == "__main__":
    organize_dataset()
