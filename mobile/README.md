# Plant Disease RAG Assistant — Mobile (Android)

On-device Android application for plant disease classification and treatment recommendations.

## Overview

This mobile version brings the Plant Disease RAG Assistant to Android devices with:
- **Offline Classification**: ViT model optimized to 82MB (INT8 quantization)
- **Local Vector Search**: sqlite-vec for fast knowledge base retrieval
- **Cloud RAG**: OpenAI API for answer synthesis (with offline fallback)
- **Target Device**: OPPO Reno8T and similar mid-range Android devices

## Architecture

```
mobile/
├── android/              # Android Studio project (to be created)
├── scripts/              # Model conversion and preparation scripts
│   ├── convert_vit_to_tflite.py
│   ├── convert_embeddings_to_tflite.py
│   └── prepare_kb_sqlite.py
├── assets/               # Converted models and data
│   ├── vit_int8.tflite
│   ├── sentence_encoder.tflite
│   └── kb.db
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.10+
- TensorFlow 2.x
- ONNX Runtime
- Android Studio (for app development)

### Phase 2: Model Conversion

```bash
# 1. Convert ViT model to TFLite (INT8)
cd mobile/scripts
python convert_vit_to_tflite.py

# 2. Convert sentence encoder to TFLite
python convert_embeddings_to_tflite.py

# 3. Prepare SQLite database with vectors
python prepare_kb_sqlite.py
```

### Phase 3: Android App Development

Coming soon: Android Studio project setup instructions.

## Performance Targets

- **Model Size**: ~82MB (ViT INT8)
- **App Size**: ~115-150MB (compressed APK)
- **Inference Latency**: <2s end-to-end
- **Accuracy**: <5% degradation vs desktop version

## Current Status

- [x] Phase 1: Research & Planning ✅
- [/] Phase 2: Model Conversion & Optimization (in progress)
- [ ] Phase 3: Knowledge Base Preparation
- [ ] Phase 4: Android App Development
- [ ] Phase 5: Testing & Optimization

## Technical Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Quantization** | INT8 | 4x size reduction, acceptable accuracy tradeoff |
| **Vector Search** | sqlite-vec | Lightweight, pure C, Android-compatible |
| **Text Generation** | OpenAI API | Proven quality, no model overhead |
| **App Architecture** | MVVM + Jetpack Compose | Modern Android best practices |

## License

Same as parent project (MIT).
