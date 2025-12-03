# Mobile ExecuTorch learnings (current state, Dec 3 2025)

## Stack alignment (working FP32)
- Working export + runtime: torch 2.10.0.dev20251025+cpu, torchvision 0.25.0.dev20251025+cpu, torchaudio 2.10.0.dev20251025+cpu, torchao 0.15.0.dev20251025+cpu, executorch 1.1.0.dev20251025+cpu.
- Android AAR: `mobile/android/app/libs/executorch-android.aar` built from the ExecuTorch repo (commit 350ea3c, debug); still compatible with the 20251025 nightly export when using the XNNPACK delegate.
- Export and on-device AAR must be from a compatible stack; otherwise FP32 can hang inside `executeNative`.

## Export steps (FP32 with XNNPACK delegate)
- Activate WSL env: `source .venv-wsl-fp32b/bin/activate`
- Command:
  ```bash
  python mobile/scripts/export_executorch_vit.py \
    --no-quantize \
    --output mobile/assets/vit_fp32_executorch.pte
  ```
- Script now always lowers with `XnnpackPartitioner()` so FP32 benefits from optimized kernels.
- On-device preprocessing matches the ViT processor: resize to 224x224, rescale to [0,1], normalize mean/std = 0.5 (see `ViTClassifier.kt`).

## Device hygiene
- Delete any forced tensor override:  
  `adb shell run-as com.example.plantdiseasemobile rm files/olive.bin`
- Ensure only the intended `.pte` is on device; the app copies the asset fresh on startup to avoid stale files.

## Quick rebuild/install
- Build: `cd mobile/android && ./gradlew assembleDebug`
- Install: `adb install -r mobile/android/app/build/outputs/apk/debug/app-debug.apk`

## Observed results
- FP32 ExecuTorch + XNNPACK now returns correct logits on multiple test images (olive peacock included) in ~1–2 seconds on device.
- SurfaceFlinger “Out of order buffers” warnings disappeared once inference latency dropped.

## Known pitfalls to avoid
- Mixing export/runtime versions (especially older executorch < 1.1) can cause missing quantized ops or hangs in FP32.
- Large FP32 model (~327 MB) takes time to push; keep INT8 handy for lighter installs if accuracy allows.
