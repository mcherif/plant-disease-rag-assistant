#!/usr/bin/env python3
"""Simple test to verify ai-edge-torch can convert a basic model"""

import torch.nn as nn
from ai_edge_torch import export_to_tflite
from pathlib import Path

# Create a simple test model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)

print("Creating simple test model...")
model = SimpleModel()
model.eval()

print("Exporting to TFLite...")
output_path = Path("mobile/test_model.tflite")
try:
    export_to_tflite(
        model=model,
        input_spec={"x": (1, 10)},
        output_path=str(output_path),
    )
    print(f"✅ Success! Model saved to {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
