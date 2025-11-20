import os
from PIL import Image
import numpy as np
from pathlib import Path

def analyze_images(directory, class_name, sample_count=10):
    """Analyze image properties for a given class"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {class_name}")
    print(f"{'='*60}")
    
    image_dir = Path(directory) / class_name
    if not image_dir.exists():
        print(f"Directory not found: {image_dir}")
        return
    
    # Get all images
    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG"))
    images = list(set(images))  # Remove duplicates
    
    print(f"Total images found: {len(images)}")
    
    if len(images) == 0:
        print("No images found!")
        return
    
    # Sample images
    sample_images = images[:min(sample_count, len(images))]
    
    resolutions = []
    modes = []
    file_sizes = []
    mean_pixels = []
    std_pixels = []
    
    for img_path in sample_images:
        try:
            img = Image.open(img_path)
            
            # Resolution
            resolutions.append(img.size)
            
            # Color mode
            modes.append(img.mode)
            
            # File size
            file_sizes.append(os.path.getsize(img_path))
            
            # Convert to numpy for stats
            img_array = np.array(img)
            
            # Pixel statistics
            mean_pixels.append(np.mean(img_array))
            std_pixels.append(np.std(img_array))
            
            img.close()
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
    
    # Summary statistics
    print(f"\nResolution Range:")
    unique_resolutions = set(resolutions)
    for res in sorted(unique_resolutions):
        count = resolutions.count(res)
        print(f"  {res[0]}x{res[1]}: {count} images")
    
    print(f"\nColor Modes:")
    for mode in set(modes):
        count = modes.count(mode)
        print(f"  {mode}: {count} images")
    
    print(f"\nFile Sizes:")
    print(f"  Min: {min(file_sizes):,} bytes ({min(file_sizes)/1024:.1f} KB)")
    print(f"  Max: {max(file_sizes):,} bytes ({max(file_sizes)/1024:.1f} KB)")
    print(f"  Avg: {np.mean(file_sizes):,.0f} bytes ({np.mean(file_sizes)/1024:.1f} KB)")
    
    print(f"\nPixel Value Statistics (0-255):")
    print(f"  Mean: {np.mean(mean_pixels):.2f} ± {np.std(mean_pixels):.2f}")
    print(f"  Std Dev: {np.mean(std_pixels):.2f} ± {np.std(std_pixels):.2f}")

# Analyze Olive classes
print("OLIVE DATASET (Kaggle)")
analyze_images("data/split/train", "Olive___healthy")
analyze_images("data/split/train", "Olive___Peacock_spot")
analyze_images("data/split/train", "Olive___Aculus_olearius")

print("\n\n")

# Compare with PlantVillage classes
print("PLANTVILLAGE DATASET (for comparison)")
analyze_images("data/split/train", "Peach___Bacterial_spot")
analyze_images("data/split/train", "Peach___healthy")
analyze_images("data/split/train", "Tomato___healthy")
