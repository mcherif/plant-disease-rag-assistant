@echo off
REM Batch script to run model conversions in conda environment

echo ============================================================
echo Plant Disease Mobile - Model Conversion
echo ============================================================
echo.

REM Activate conda environment
call conda activate plant-disease-mobile
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment
    exit /b 1
)

echo Environment activated: plant-disease-mobile
echo.

REM Step 1: Convert ViT model
echo [1/3] Converting ViT model to TFLite INT8...
python convert_vit_to_tflite.py --input ..\..\models\vit-finetuned --output ..\assets\vit_int8.tflite
if errorlevel 1 (
    echo ERROR: ViT conversion failed
    exit /b 1
)
echo.

REM Step 2: Convert embedding model
echo [2/3] Converting embedding model to TFLite...
python convert_embeddings_to_tflite.py --model all-MiniLM-L6-v2 --output ..\assets\sentence_encoder.tflite
if errorlevel 1 (
    echo ERROR: Embedding conversion failed
    exit /b 1
)
echo.

REM Step 3: Prepare knowledge base
echo [3/3] Preparing knowledge base SQLite database...
python prepare_kb_sqlite.py --kb ..\..\data\kb --output ..\assets\kb.db --test
if errorlevel 1 (
    echo ERROR: KB preparation failed
    exit /b 1
)
echo.

echo ============================================================
echo All conversions completed successfully!
echo ============================================================
echo.
echo Output files:
dir /b ..\assets\*.tflite ..\assets\*.db 2>nul
echo.

pause
