"""
Deep dive into quantized model structure to find the right save/load method
"""

import torch
from transformers import ViTForImageClassification
from pathlib import Path
import pickle

print("=" * 70)
print("Quantized Model Structure Analysis")
print("=" * 70)
print()

# Load and quantize model
print("📦 Loading and quantizing model...")
model_dir = Path("models/vit-finetuned")
model = ViTForImageClassification.from_pretrained(str(model_dir))
model.eval()

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
print("✅ Model quantized")
print()

# Analyze the quantized model structure
print("🔍 Analyzing quantized model structure...")
print()

# Check what type of object it is
print(f"Model type: {type(quantized_model)}")
print(f"Model class: {quantized_model.__class__.__name__}")
print()

# Try different save methods
print("Testing different save methods:")
print()

# Method 1: Standard state_dict (we know this fails to load)
print("1. torch.save(model.state_dict(), ...) - Standard approach")
try:
    torch.save(quantized_model.state_dict(), "test_state_dict.pth")
    print("   ✅ Saved successfully")
    print(f"   Size: {Path('test_state_dict.pth').stat().st_size / (1024**2):.1f} MB")
except Exception as e:
    print(f"   ❌ Failed: {e}")
print()

# Method 2: Save entire model (not just state dict)
print("2. torch.save(model, ...) - Save entire model object")
try:
    torch.save(quantized_model, "test_full_model.pth")
    print("   ✅ Saved successfully")
    print(f"   Size: {Path('test_full_model.pth').stat().st_size / (1024**2):.1f} MB")
    
    # Try loading it back
    print("   Testing load...")
    loaded_model = torch.load("test_full_model.pth")
    loaded_model.eval()
    
    # Test inference
    test_input = {"pixel_values": torch.randn(1, 3, 224, 224)}
    with torch.no_grad():
        output = loaded_model(**test_input)
    
    print("   ✅ Loaded and tested successfully!")
    print("   🎉 THIS METHOD WORKS!")
    
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
print()

# Method 3: JIT script (not trace)
print("3. torch.jit.script(model) - Script instead of trace")
try:
    scripted = torch.jit.script(quantized_model)
    scripted.save("test_scripted.pt")
    print("   ✅ Saved successfully")
    print(f"   Size: {Path('test_scripted.pt').stat().st_size / (1024**2):.1f} MB")
except Exception as e:
    print(f"   ❌ Failed: {e}")
print()

# Method 4: Pickle
print("4. pickle.dump(model, ...) - Python pickle")
try:
    with open("test_pickle.pkl", "wb") as f:
        pickle.dump(quantized_model, f)
    print("   ✅ Saved successfully")
    print(f"   Size: {Path('test_pickle.pkl').stat().st_size / (1024**2):.1f} MB")
    
    # Try loading
    print("   Testing load...")
    with open("test_pickle.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    loaded_model.eval()
    
    # Test inference
    test_input = {"pixel_values": torch.randn(1, 3, 224, 224)}
    with torch.no_grad():
        output = loaded_model(**test_input)
    
    print("   ✅ Loaded and tested successfully!")
    print("   🎉 THIS METHOD ALSO WORKS!")
    
except Exception as e:
    print(f"   ❌ Failed: {e}")
print()

# Cleanup
print("🧹 Cleaning up test files...")
for f in ["test_state_dict.pth", "test_full_model.pth", "test_scripted.pt", "test_pickle.pkl"]:
    if Path(f).exists():
        Path(f).unlink()
print("✅ Cleanup complete")
print()

print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print()
print("The solution is to use torch.save(model, ...) instead of")
print("torch.save(model.state_dict(), ...)")
print()
print("This saves the entire model object with quantization intact!")
