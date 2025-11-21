# Model Conversion - Installation & Troubleshooting

## Issue: Conda Environment DLL Conflict

The `plant-disease-mobile` conda environment has a PyTorch DLL conflict on Windows. 

**Solution**: Use your existing Python environment and install only the missing packages.

## Install Missing Dependencies

```bash
# Check what you have
python -c "import torch; print('✅ PyTorch:', torch.__version__)"

# Install missing packages
pip install tensorflow>=2.13.0
pip install onnx>=1.14.0
pip install onnx-tf>=1.10.0

# Verify installation
python -c "import tensorflow; print('✅ TensorFlow:', tensorflow.__version__)"
python -c "import onnx; print('✅ ONNX:', onnx.__version__)"
```

## Run Conversions

Once dependencies are installed:

```bash
cd mobile/scripts

# 1. Convert ViT model (5-10 minutes)
python convert_vit_to_tflite.py

# 2. Convert embedding model (3-5 minutes)
python convert_embeddings_to_tflite.py

# 3. Prepare knowledge base (2-3 minutes)
python prepare_kb_sqlite.py --test
```

## Expected Output

After successful conversion, you should have:
- `mobile/assets/vit_int8.tflite` (~82MB)
- `mobile/assets/sentence_encoder.tflite` (~25MB)
- `mobile/assets/kb.db` (~5-10MB)

## Alternative: Skip Conda Environment

You can delete the problematic conda environment:
```bash
conda env remove -n plant-disease-mobile
```

The main project environment works fine - just install the 3 missing packages above.
