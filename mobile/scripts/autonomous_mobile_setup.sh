#!/bin/bash
# Autonomous mobile model quantization script
# Tries multiple approaches to get a working quantized model

set -e

echo "============================================================"
echo "Mobile Model Quantization - Autonomous Attempt"
echo "============================================================"
echo ""

cd /mnt/c/projects/plant-disease-rag-assistant/mobile/scripts
source venv/bin/activate

# Approach 1: ONNX Runtime Static Quantization
echo "[Approach 1] ONNX Runtime Static Quantization..."
python quantize_onnx_model.py --input ../assets/vit_model.onnx --output ../assets/vit_int8.onnx 2>&1 | tee quantization_log.txt

if [ -f ../assets/vit_int8.onnx ]; then
    echo "✅ Quantization successful!"
    ls -lh ../assets/vit_int8.onnx
else
    echo "❌ Quantization failed, see quantization_log.txt"
fi

echo ""
echo "Creating mobile SQLite database..."
python prepare_kb_sqlite.py --kb ../../data/kb --output ../assets/kb.db --test

echo ""
echo "============================================================"
echo "Autonomous work complete!"
echo "============================================================"
