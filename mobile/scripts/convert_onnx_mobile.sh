#!/bin/bash
# Updated WSL conversion script using ONNX Runtime Mobile approach

set -e

echo "============================================================"
echo "Plant Disease Mobile - ONNX Model Preparation"
echo "============================================================"
echo ""

cd /mnt/c/projects/plant-disease-rag-assistant/mobile/scripts

# Activate venv
source venv/bin/activate

echo "[1/4] Installing ONNX Runtime..."
pip install onnxruntime

echo ""
echo "[2/4] Quantizing ONNX model to INT8..."
python quantize_onnx_model.py --input ../assets/vit_model.onnx --output ../assets/vit_int8.onnx

echo ""
echo "[3/4] Converting embedding model..."
python convert_embeddings_to_tflite.py --model all-MiniLM-L6-v2 --output ../assets/sentence_encoder.tflite

echo ""
echo "[4/4] Preparing knowledge base..."
python prepare_kb_sqlite.py --kb ../../data/kb --output ../assets/kb.db --test

echo ""
echo "============================================================"
echo "All conversions completed!"
echo "============================================================"
echo ""
echo "Output files:"
ls -lh ../assets/ | grep -E '\.(onnx|tflite|db)$'
echo ""
echo "Mobile deployment approach: ONNX Runtime Mobile"
echo "- ViT model: vit_int8.onnx (~82MB, INT8 quantized)"
echo "- Embedding: sentence_encoder.tflite (~25MB)"
echo "- Knowledge base: kb.db (~5-10MB)"
echo ""
