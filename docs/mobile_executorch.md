## Mobile ExecuTorch Quantization & Validation

### Artifacts
- Quantized model: `mobile/assets/vit_int8_executorch_qnnpack.pte` (~82.5 MB, XNNPACK/CPU).
- Export script: `mobile/scripts/export_executorch_vit_qnnpack.py`.
- Test helpers: `mobile/scripts/test_executorch_vit_qnnpack.py`, `mobile/scripts/run_executorch_vit_qnnpack.py`.

### Build ExecuTorch with runtime (WSL)
```bash
cd /mnt/c/projects/executorch
python3 -m venv .venv-build && source .venv-build/bin/activate
pip install --upgrade pip cmake ninja pillow transformers==4.57.1 \
    --index-url https://download.pytorch.org/whl/nightly/cpu "torch==2.10.0.dev20251122+cpu"
pip install pyyaml

git submodule update --init --recursive
mkdir -p build && cd build
/mnt/c/projects/executorch/.venv-build/bin/cmake .. -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DEXECUTORCH_BUILD_RUNTIME=ON \
  -DEXECUTORCH_BUILD_PYTHON_BINDINGS=ON \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_PYBIND=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DPYTHON_EXECUTABLE=/mnt/c/projects/executorch/.venv-build/bin/python
ninja
pip install --no-build-isolation .
```

### Desktop validation (quantized .pte)
```bash
# ExecuTorch runtime installed in .venv-build
source /mnt/c/projects/executorch/.venv-build/bin/activate
cd /mnt/c/projects/plant-disease-rag-assistant
/mnt/c/projects/executorch/build/executor_runner \
  --model_path=mobile/assets/vit_int8_executorch_qnnpack.pte \
  --num_executions=1 \
  --print_all_output    # fills inputs with ones
```
Expected: prints logits; confirms the quantized model executes on CPU via XNNPACK.

### Exporting the model
```bash
source .venv-exec-nightly/bin/activate  # or another env with torch>=2.9+executorch+torchao
python mobile/scripts/export_executorch_vit_qnnpack.py
```
Outputs: `mobile/assets/vit_int8_executorch_qnnpack.pte` with timing info.

### Running over an image folder
```bash
source .venv-exec-nightly/bin/activate  # requires transformers+Pillow
python mobile/scripts/run_executorch_vit_qnnpack.py
```
Writes `mobile/assets/predictions_qnnpack.csv`. If ExecuTorch runtime isn’t present in the env, it falls back to FP32 and logs it loudly.

### Android notes
- The `.pte` uses the XNNPACK delegate → CPU-only on device.
- To try NPU/GPU, re-export for an NNAPI/QNN backend and test on-device; support varies by SoC (Snapdragon 695/Helio G99 likely stay on CPU).
