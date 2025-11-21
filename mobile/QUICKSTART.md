# Mobile Deployment - Quick Start Guide

## Phase 2: Model Conversion

### Prerequisites

```bash
# Navigate to mobile scripts directory
cd mobile/scripts

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 1: Convert ViT Model to TFLite

```bash
python convert_vit_to_tflite.py \
    --input ../../models/vit-finetuned \
    --output ../assets/vit_int8.tflite
```

**Expected output:**
- `vit_int8.tflite` (~82MB)
- Conversion time: ~5-10 minutes
- Accuracy validation report

### Step 2: Convert Embedding Model to TFLite

```bash
python convert_embeddings_to_tflite.py \
    --model all-MiniLM-L6-v2 \
    --output ../assets/sentence_encoder.tflite
```

**Expected output:**
- `sentence_encoder.tflite` (~20-25MB)
- Conversion time: ~3-5 minutes

### Step 3: Prepare Knowledge Base

```bash
python prepare_kb_sqlite.py \
    --kb ../../data/kb \
    --output ../assets/kb.db \
    --test
```

**Expected output:**
- `kb.db` (~5-10MB with embeddings)
- Vector search test results

### Troubleshooting

**Issue: `onnx-tf` import error**
```bash
pip install onnx-tf
```

**Issue: TensorFlow not found**
```bash
pip install tensorflow>=2.13.0
```

**Issue: Out of memory during conversion**
- Reduce batch size in representative dataset
- Use smaller calibration dataset (50 samples instead of 100)

## Next Steps

After successful conversion:
1. Verify all files in `mobile/assets/`:
   - `vit_int8.tflite` (~82MB)
   - `sentence_encoder.tflite` (~25MB)
   - `kb.db` (~5-10MB)

2. Proceed to Phase 3: Android app development

## Notes

- Conversion scripts use representative datasets for INT8 calibration
- For production, use actual validation images for better accuracy
- TFLite models are optimized for Android NNAPI acceleration
