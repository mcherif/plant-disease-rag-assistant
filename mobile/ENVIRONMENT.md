  # Mobile Environment Setup

## Option 1: Conda Environment (Recommended)

```bash
# Navigate to mobile directory
cd mobile

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate plant-disease-mobile

# Verify installation
python -c "import torch; import tensorflow; import onnx; print('✅ All dependencies installed')"
```

## Option 2: Python venv

```bash
cd mobile/scripts
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Details

**Name:** `plant-disease-mobile`

**Python Version:** 3.10

**Key Dependencies:**
- PyTorch 2.1+ (for loading ViT model)
- TensorFlow 2.13+ (for TFLite conversion)
- ONNX + onnx-tf (for intermediate conversion)
- Transformers (for HuggingFace models)
- sentence-transformers (for embedding model)

**Size:** ~5-7GB (includes PyTorch, TensorFlow, and CUDA if available)

## Quick Start After Setup

```bash
# Activate environment
conda activate plant-disease-mobile

# Run conversions
cd scripts
python convert_vit_to_tflite.py
python convert_embeddings_to_tflite.py
python prepare_kb_sqlite.py --test
```

## Deactivate When Done

```bash
conda deactivate
```

## Remove Environment (if needed)

```bash
conda env remove -n plant-disease-mobile
```
