---
title: Plant-Disease-RAG-Assistant
emoji: 🌿
pinned: false
short_description: RAG-powered assistant for plant disease diagnosis & guidance
license: mit
tags: ["plant", "plant-disease", "plantvillage", "rag", "llm", "retrieval", "bm25", "vector-search", "streamlit", "gradio", "fastapi"]
sdk: streamlit
app_file: src/interface/streamlit_app.py
app_port: 7860
---

<!-- Logo/banner at top -->
<p align="center">
  <img src="images/plant-disease-rag-assistant-logo.png" alt="Plant Disease RAG Assistant Logo" width="300"/>
</p>

# 🌿 Plant-Disease-RAG-Assistant

A Retrieval-Augmented Generation (RAG) assistant for plant disease diagnosis and guidance. It leverages Vision Transformer (ViT) image classification, hybrid retrieval (BM25 + vector search), and LLM-powered Q&A to provide grounded answers from curated sources. The system supports **15 crops** with **41+ disease classes** and features GPU acceleration for faster inference.

---

## 🚀 Try the App Online

**[🌿 Plant Disease RAG Assistant on Hugging Face Spaces](https://huggingface.co/spaces/mcherif/Plant-Disease-RAG-Assistant)**

No setup required—just open the link and start exploring!

---

## ✨ Features

- **🔬 Image Classification**: Diagnose plant diseases from uploaded images using a fine-tuned Vision Transformer (ViT) model
- **🌾 15 Supported Crops**: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato, Pepper (bell), and **Olive**
- **🦠 41+ Disease Classes**: Comprehensive coverage of common plant diseases
- **📚 Document Retrieval**: Search and retrieve relevant information from a curated knowledge base
- **💬 Conversational Assistant**: Ask questions and receive context-aware, grounded answers
- **🔄 Hybrid Retrieval**: Combines BM25 and FAISS vector search for robust results
- **⚡ GPU Acceleration**: CUDA support for faster inference on compatible hardware
- **🔧 Resilient Indexing**: Automatically rebuilds FAISS index when needed (helpful for CI/CD)
- **🌐 Multiple Interfaces**: Streamlit (main UI), FastAPI (REST API), Gradio (deprecated)
- **📊 Feedback & Monitoring**: Built-in feedback collection and dashboard

---

## 🌱 Supported Crops

This project currently supports **15 crops** and **41+ disease classes**:

| Crop | Icon | Disease Count |
|------|------|---------------|
| Apple | 🍎 | 4+ diseases |
| Blueberry | 🫐 | 1 class |
| Cherry (including sour) | 🍒 | 2+ diseases |
| Corn (maize) | 🌽 | 4+ diseases |
| Grape | 🍇 | 4+ diseases |
| Orange | 🍊 | 1+ disease |
| Peach | 🍑 | 2+ diseases |
| Potato | 🥔 | 3+ diseases |
| Raspberry | 🫐 | 1 class |
| Soybean | 🌱 | 1 class |
| Squash | 🥒 | 1+ disease |
| Strawberry | 🍓 | 2+ diseases |
| Tomato | 🍅 | 10+ diseases |
| Pepper, bell | 🫑 | 2+ diseases |
| **Olive** | 🫒 | **3+ diseases** ✨ |

For a complete breakdown of diseases per crop, see [`plant_diseases_table.csv`](plant_diseases_table.csv).

---

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
docker compose up --build
```

- **Streamlit UI**: http://localhost:8501
- **Gradio UI** (deprecated): http://localhost:7860

> **Note**: To enable LLM-powered answers, set your OpenAI API key:
> ```bash
> docker compose up --build -e OPENAI_API_KEY=sk-...
> ```

### Option 2: Local Development with Conda (GPU Support)

For optimal performance with CUDA GPU acceleration:

```bash
# Create and activate conda environment with GPU support
conda create -n plant-disease-gpu python=3.11
conda activate plant-disease-gpu

# Install PyTorch with CUDA support (for NVIDIA GPUs)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install project dependencies
pip install -r requirements.txt

# Set your OpenAI API key (required for LLM answers)
export OPENAI_API_KEY=sk-...  # Linux/Mac
# OR
$env:OPENAI_API_KEY="sk-..."  # Windows PowerShell

# Run Streamlit app
streamlit run src/interface/streamlit_app.py
```

**Device Selection**: The Streamlit app automatically detects CUDA availability. Select "cuda" in the Device dropdown for GPU acceleration, or "cpu" for CPU-only inference.

### Option 3: Local Development (CPU Only)

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY=sk-...  # Linux/Mac
# OR
$env:OPENAI_API_KEY="sk-..."  # Windows PowerShell

# Run Streamlit app
streamlit run src/interface/streamlit_app.py
```

### Mobile quantized model (ExecuTorch)

- ExecuTorch/XNNPACK quantized ViT (~82 MB `.pte`) for on-device CPU inference.
- See `docs/mobile_executorch.md` for export steps, desktop validation, and Android integration notes.
- See `docs/mobile_android_app.md` for a minimal on-device demo app (Compose UI, ExecuTorch).

### Mobile FP32 (ExecuTorch + XNNPACK)

- Working FP32 export + runtime stack (nightly 20251025): torch 2.10.0.dev20251025+cpu, executorch 1.1.0.dev20251025+cpu (see `docs/mobile_env_fp32.txt`).
- Export command: `python mobile/scripts/export_executorch_vit.py --no-quantize --output mobile/assets/vit_fp32_executorch.pte`
- The export script now always lowers with the XNNPACK delegate for speed; on-device runs take ~1–2s on the olive test image.
- Lessons and pitfalls recorded in `docs/mobile_learnings.md` (stack alignment, forced tensor removal, build/install steps).

### Install the Android APK via adb

```bash
# From repo root
adb install -r mobile/android/app/build/outputs/apk/debug/app-debug.apk
```
If you are on Windows PowerShell:
```powershell
& "$Env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" install -r "mobile/android/app/build/outputs/apk/debug/app-debug.apk"
```
- After `./gradlew assembleDebug`, a copy is saved as `mobile/android/app/build/outputs/apk/debug/plant-disease-mobile-debug.apk` for easier sharing. The build log prints the exact path.
- To build a release APK: `cd mobile/android && ./gradlew assembleRelease` (configure signing first). The artifact will be under `app/build/outputs/apk/release/`.
- After `assembleRelease`, a friendly copy is saved as `mobile/android/app/build/outputs/apk/release/plant-disease-mobile-release.apk`. This is the signed release you can distribute.
- To generate a QR for a hosted APK URL: `pip install qrcode[pil]` then `python scripts/generate_apk_qr.py --url https://your-hosted-url/plant-disease-mobile-release.apk`.

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t plant-disease-rag-assistant .
```

### Run with Environment Variables

```bash
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=sk-... \
  plant-disease-rag-assistant
```

This launches the Streamlit UI at [http://localhost:8501](http://localhost:8501).

---

## 📂 Project Structure

```
plant-disease-rag-assistant/
├── README.md                  # Project overview and quick start
├── requirements*.txt          # Dependencies (core + dev)
├── docker-compose.yml         # Local orchestration
├── Makefile                   # Common commands
├── data/                      # Datasets and KB artifacts
├── models/                    # Fine-tuned ViT and indexes
├── mobile/                    # Mobile exports/scripts/assets (ExecuTorch, ONNX, etc.)
│   └── assets/                # Quantized models (e.g., vit_int8_executorch_qnnpack.pte)
├── notebooks/                 # EDA and experimentation
├── src/                       # App, classifier, ingestion, retrieval, LLM, interfaces
├── scripts/                   # Utility scripts
├── tests/                     # Unit tests
└── docs/                      # Documentation (incl. docs/mobile_executorch.md)
```

### Knowledge base and hyperlinks

- KB artifacts live in `data/kb/` (chunks + manifest) and the FAISS index in `models/index/kb-faiss-bge/`.
- The Streamlit “Sources” list renders clickable links when `meta.jsonl` contains non-empty URLs.
- To rebuild the PlantVillage KB (with URLs) and refresh the index:
  ```bash
  python -m src.ingestion.build_kb --sources plantvillage --out data/kb --min_tokens 20 --max_tokens 400 --overlap 40 --dedup none --verbose
  python -m src.retrieval.build_index --manifest data/kb/manifest.parquet --out models/index/kb-faiss-bge --model BAAI/bge-small-en-v1.5 --device cpu --batch-size 8
  ```
- Restart Streamlit after rebuilding so it picks up the updated metadata and renders hyperlinks.

---

## 🌐 API Endpoints

The FastAPI backend exposes the following RESTful endpoints:

### Health Check
- **GET `/health`**  
  Returns API status, configuration, and metadata (index path, device, top_k, document count, model info)

### RAG Question Answering
- **POST `/rag`**  
  Accepts a user query and optional filters (plant, disease, top_k, fusion, alpha, temperature, timeout)  
  Returns the LLM-generated answer, sources, retrieved chunks, and latency

### Image Classification
- **POST `/api/classify`**  
  Accepts an image file upload (plant leaf)  
  Returns top predicted disease labels and confidence scores using the ViT model

### Feedback Collection
- **POST `/api/feedback`**  
  Accepts feedback about answers (thumbs up/down, comments, etc.) as JSON  
  Stores feedback for monitoring and dashboarding

Example payloads for each endpoint are available in the code comments.

---

## 💻 Usage

### Streamlit Web UI (Recommended)

The main interactive interface for plant disease diagnosis and Q&A:

```bash
# With conda (GPU support)
conda activate plant-disease-gpu
streamlit run src/interface/streamlit_app.py

# OR with venv (CPU only)
.venv\Scripts\activate
streamlit run src/interface/streamlit_app.py
```

**Features**:
- Upload or capture plant images for classification
- View top-3 disease predictions with confidence scores
- Ask questions about detected diseases
- Browse sources and provide feedback
- Select CPU or CUDA device (auto-detected)

See [docs/STREAMLIT.md](docs/STREAMLIT.md) for detailed UI documentation.

### FastAPI Backend

Run the REST API server for programmatic access:

```bash
uvicorn src.interface.api:app --host 0.0.0.0 --port 8000
```

Access API docs at: http://localhost:8000/docs

### Gradio UI (Deprecated)

For legacy compatibility only (use Streamlit for new features):

```bash
python src/interface/app_gradio.py
```

---

## 📸 Sample App Display

<p align="center">
  <img src="images/sample_screenshot_PlantDiseaseRAGAssistant.png" alt="Sample Plant Disease RAG Assistant UI" width="700"/>
</p>

**Example workflow:**
1. Upload a plant image (e.g., olive leaf with peacock spot)
2. View top predictions with disease names and confidence scores  
3. Enter a question or use detected plant/disease for context-aware Q&A
4. See the LLM-generated answer with citations
5. Provide feedback to improve the system

---

## 📦 Deployment Options

### Hugging Face Spaces (Public Demo)

**[🌿 Try it online here](https://huggingface.co/spaces/mcherif/Plant-Disease-RAG-Assistant)** - No setup required!

### FastAPI Service (Production)

Deploy the FastAPI backend for custom workflows, integrations, and automation:

```bash
# Production server with multiple workers
uvicorn src.interface.api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Swarm / Kubernetes

Use the provided `Dockerfile` and `docker-compose.yml` as templates for orchestrated deployments.

---

## 📚 Dataset & Knowledge Base

### Sources

The knowledge base is built from:
- **PlantVillage**: Disease descriptions, symptoms, and management
- **Wikipedia**: Additional plant disease articles
- **Custom curation**: Manually verified and cleaned entries

See [docs/data_card.md](docs/data_card.md) for detailed sources, build steps, and limitations.

### Expanding Disease Coverage

To add new crops or diseases:

1. **Update seed pairs** in [`data/plantvillage_kb.json`](data/plantvillage_kb.json)
2. **Extend normalization** in [`src/ingestion/build_kb.py`](src/ingestion/build_kb.py)
3. **Rebuild KB** and re-index:

```bash
python -m src.ingestion.build_kb \
  --sources plantvillage,wikipedia \
  --out data/kb \
  --min_tokens 50 \
  --max_tokens 400 \
  --overlap 80 \
  --dedup minhash \
  --dedup-threshold 0.9 \
  --wiki-lang en \
  --wiki-interval 0.5 \
  --verbose
```

4. **Regenerate crop list**:

```bash
python scripts/list_all_plant_diseases_covered.py
```

This will update `plant_diseases_table.csv` with the latest crops and diseases.

---

## 🔍 Retrieval System

### Hybrid Retrieval Pipeline

- **BM25**: Keyword-based sparse retrieval for exact matches
- **FAISS**: Dense vector search using BGE embeddings (all-MiniLM-L6-v2 or similar)
- **Reciprocal Rank Fusion**: Combines BM25 and FAISS results for best-of-both-worlds retrieval

### FAISS Fallback (CI/CD Support)

Retrieval tests succeed even when binary FAISS artifacts are missing (e.g., on GitHub Actions or fresh git-lfs checkouts). The pipeline automatically rebuilds an in-memory index from `data/kb/kb.json` when needed.

Control this behavior with the `RAG_ALLOW_FAISS_FALLBACK` environment variable (defaults to `1`).

See [docs/retrieval.md](docs/retrieval.md) for detailed retrieval documentation and [docs/artifacts.md](docs/artifacts.md) for evaluation results.

---

## 🧪 Testing

Run the test suite:

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run specific test file
pytest tests/test_retrieval.py

# Run with verbose output
pytest -v
```

See [docs/testing.md](docs/testing.md) for testing, OpenAI integration, and debugging tips.

---

## 🛠️ Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key for LLM responses | - | Yes (for Q&A) |
| `OPENAI_MODEL` | OpenAI model to use | `gpt-4o-mini` | No |
| `RAG_ALLOW_FAISS_FALLBACK` | Allow rebuilding FAISS index from KB | `1` | No |

---

## 📋 Requirements

### Core Dependencies

- Python 3.10+
- PyTorch >= 2.1.0 (CPU or CUDA)
- transformers (for ViT model)
- sentence-transformers (for embeddings)
- faiss-cpu or faiss-gpu (for vector search)
- openai >= 1.30.0
- streamlit (for UI)
- fastapi + uvicorn (for API)

See [`requirements.txt`](requirements.txt) for the complete list.

### Optional: GPU Support

For CUDA GPU acceleration:
- NVIDIA GPU with CUDA 12.1+ support
- PyTorch with CUDA (install via conda as shown in Quick Start)

---

## 🗺️ Roadmap

See [docs/improvements.md](docs/improvements.md) for prioritized backlog, feature ideas, and test plans.

**Upcoming**:
- [ ] Mobile app deployment (ExecuTorch/NNAPI experiments)
- [ ] Multilingual support
- [ ] Real-time disease monitoring
- [ ] Integration with agricultural IoT sensors
- [ ] Expanded crop coverage (30+ crops)

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **PlantVillage** for disease information and imagery
- **Hugging Face** for model hosting and Spaces platform
- **OpenAI** for LLM capabilities
- Community contributors and testers

---

## 📧 Contact & Support

- **Author**: Mohamed Cherif
- **Email**: innerloopinc@gmail.com
- **GitHub**: [mcherif/plant-disease-rag-assistant](https://github.com/mcherif/plant-disease-rag-assistant)
- **Issues**: [Report bugs or request features](https://github.com/mcherif/plant-disease-rag-assistant/issues)

---

**⭐ If you find this project helpful, please star it on GitHub!**
