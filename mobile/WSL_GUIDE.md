# WSL Model Conversion Guide

## Quick Start

### 1. Open WSL Terminal

```bash
# Open WSL (Ubuntu) from Windows Terminal or PowerShell
wsl
```

### 2. Navigate to Project

```bash
cd /mnt/c/projects/plant-disease-rag-assistant/mobile/scripts
```

### 3. Run Conversion Script

```bash
# Make script executable
chmod +x setup_and_convert.sh

# Run it
bash setup_and_convert.sh
```

This will:
- Create Python virtual environment
- Install all dependencies (torch, tensorflow, onnx, etc.)
- Convert ViT model to TFLite INT8
- Convert embedding model to TFLite
- Prepare SQLite knowledge base
- Test everything

**Expected time:** 10-15 minutes (first run with downloads)

---

## Manual Step-by-Step (if script fails)

### Setup Environment

```bash
cd /mnt/c/projects/plant-disease-rag-assistant/mobile/scripts

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision transformers tensorflow onnx onnx-tf pillow numpy pandas sentence-transformers
```

### Run Conversions

```bash
# 1. ViT model (5-10 minutes)
python convert_vit_to_tflite.py

# 2. Embedding model (3-5 minutes)
python convert_embeddings_to_tflite.py

# 3. Knowledge base (2-3 minutes)
python prepare_kb_sqlite.py --test
```

### Verify Output

```bash
ls -lh ../assets/
# Should see:
# - vit_int8.tflite (~82MB)
# - sentence_encoder.tflite (~25MB)
# - kb.db (~5-10MB)
```

---

## Troubleshooting

### WSL Not Installed?

```powershell
# In PowerShell (as Administrator)
wsl --install
# Restart computer
```

### Python3 Not Found?

```bash
# In WSL
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### Permission Issues?

```bash
# Make sure you're in the right directory
pwd
# Should show: /mnt/c/projects/plant-disease-rag-assistant/mobile/scripts

# Check file access
ls -la ../../models/vit-finetuned/
```

### Out of Disk Space?

```bash
# Check WSL disk usage
df -h

# If needed, clean up
sudo apt clean
pip cache purge
```

---

## After Conversion

Files will be accessible from Windows at:
```
C:\projects\plant-disease-rag-assistant\mobile\assets\
```

You can then proceed to Phase 3: Android app development!
