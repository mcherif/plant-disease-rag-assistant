#!/bin/bash
# WSL Setup and Conversion Script for Plant Disease Mobile
# Run this in WSL: bash setup_and_convert.sh

set -e  # Exit on error

echo "============================================================"
echo "Plant Disease Mobile - WSL Model Conversion"
echo "============================================================"
echo ""

# Navigate to project directory (WSL can access Windows files via /mnt/c/)
PROJECT_DIR="/mnt/c/projects/plant-disease-rag-assistant/mobile/scripts"
cd "$PROJECT_DIR"

echo "Current directory: $(pwd)"
echo ""

# Step 1: Create Python virtual environment
echo "[1/5] Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi
echo ""

# Step 2: Activate environment and install dependencies
echo "[2/5] Installing dependencies..."
source venv/bin/activate

pip install --upgrade pip
pip install torch torchvision transformers tensorflow onnx onnx-tf pillow numpy pandas sentence-transformers

echo "✅ Dependencies installed"
echo ""

# Step 3: Convert ViT model
echo "[3/5] Converting ViT model to TFLite INT8..."
python convert_vit_to_tflite.py --input ../../models/vit-finetuned --output ../assets/vit_int8.tflite

if [ $? -eq 0 ]; then
    echo "✅ ViT conversion successful"
else
    echo "❌ ViT conversion failed"
    exit 1
fi
echo ""

# Step 4: Convert embedding model
echo "[4/5] Converting embedding model to TFLite..."
python convert_embeddings_to_tflite.py --model all-MiniLM-L6-v2 --output ../assets/sentence_encoder.tflite

if [ $? -eq 0 ]; then
    echo "✅ Embedding conversion successful"
else
    echo "❌ Embedding conversion failed"
    exit 1
fi
echo ""

# Step 5: Prepare knowledge base
echo "[5/5] Preparing knowledge base SQLite database..."
python prepare_kb_sqlite.py --kb ../../data/kb --output ../assets/kb.db --test

if [ $? -eq 0 ]; then
    echo "✅ KB preparation successful"
else
    echo "❌ KB preparation failed"
    exit 1
fi
echo ""

# Summary
echo "============================================================"
echo "All conversions completed successfully!"
echo "============================================================"
echo ""
echo "Output files:"
ls -lh ../assets/*.tflite ../assets/*.db 2>/dev/null || echo "Checking files..."
echo ""
echo "Next steps:"
echo "1. Files are in: C:\\projects\\plant-disease-rag-assistant\\mobile\\assets"
echo "2. Proceed to Android app development (Phase 3)"
echo ""
