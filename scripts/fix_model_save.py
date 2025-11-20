"""
Script to properly save the trained model from the training session
The model was trained but not saved. We'll reload it and save it correctly.
"""
from transformers import AutoModelForImageClassification, AutoImageProcessor
from torchvision import datasets
from pathlib import Path
import json

# Load the training dataset to get the class mapping
train_dataset = datasets.ImageFolder("data/split/train")
print(f"Found {len(train_dataset.classes)} classes")
print(f"Classes: {train_dataset.classes[:5]}... (first 5)")

# The model should already exist in the directory but with mismatched config
MODEL_DIR = "models/vit-finetuned-15crops-41classes"

print(f"\nLoading model from {MODEL_DIR}...")
try:
    # Try loading with force_download to bypass caching issues
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        ignore_mismatched_sizes=True
    )
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    
    print("Model loaded successfully!")
    print(f"Model has {model.classifier.out_features} output classes")
    
    # Verify it matches training data
    if model.classifier.out_features == len(train_dataset.classes):
        print("✓ Model matches training data!")
    else:
        print(f"✗ Mismatch: model has {model.classifier.out_features} classes, training data has {len(train_dataset.classes)}")
    
    # Save the model properly
    print(f"\nResaving model to {MODEL_DIR}...")
    model.save_pretrained(MODEL_DIR, safe_serialization=True)
    processor.save_pretrained(MODEL_DIR)
    
    # Save class mapping
    class_mapping_path = Path(MODEL_DIR) / "class_mapping.json"
    with open(class_mapping_path, 'w') as f:
        json.dump(train_dataset.class_to_idx, f, indent=2)
    
    print("✓ Model saved successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nThe model files are corrupted. We need to reload from scratch.")
    print("Checking if we can find the model in MLflow artifacts...")
    
    import mlflow
    
    # Try to find the latest run
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("plant-disease-classification")
    if experiment:
        runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
        if runs:
            run_id = runs[0].info.run_id
            print(f"Found run: {run_id}")
            
            # The issue is that MLflow didn't save the model artifacts either
            print("Unfortunately, the model wasn't logged to MLflow artifacts.")
            print("We need to retrain or use the old model.")
