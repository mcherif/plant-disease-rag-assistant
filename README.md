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

A Retrieval-Augmented Generation (RAG) assistant for plant disease diagnosis and guidance. It leverages classic (BM25) and vector-based retrieval, orchestrated with an LLM, to provide grounded answers from curated sources (docs, PDFs, web pages). The system supports image classification, document search, and conversational Q&A.

---

## 🚀 Try the App Online

**[🌿 Plant Disease RAG Assistant on Hugging Face Spaces](https://huggingface.co/spaces/mcherif/Plant-Disease-RAG-Assistant)**

No setup required—just open the link and start exploring!

---

## Features

- **Image classification**: Diagnose plant diseases from uploaded images.
- **Document retrieval**: Search and retrieve relevant information from a knowledge base.
- **Conversational assistant**: Ask questions and receive context-aware, grounded answers.
- **Hybrid retrieval**: Combines BM25 and vector search (FAISS) for robust results.
- **APIs**: RESTful endpoints for integration and automation.
- **Multi-UI support**: Streamlit (main), FastAPI backend, Gradio (deprecated)

## Quick Start

### Docker Compose (recommended)
```powershell
docker compose up --build
```
- Streamlit UI: http://localhost:8501
- Gradio UI (deprecated): http://localhost:7860

### Plain Docker
```powershell
docker build -t plant-llm-assistant .
docker run -p 7860:7860 -p 8501:8501 -v ${PWD}:/code plant-llm-assistant
```

## 🐳 Build and Run with Docker

To build the Docker image for this app, run:

```bash
docker build -t plant-disease-rag-assistant .
```

To start the app in a container:

```bash
docker run -p 8501:8501 plant-disease-rag-assistant
```

Or, if you use Docker Compose:

```bash
docker compose up --build
```

> **Note:** To enable LLM-powered answers, set your OpenAI API key as an environment variable:
>
> ```bash
> docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... plant-disease-rag-assistant
> ```

## 🐳 Run the Docker Container

To start the app in a Docker container, use:

```bash
docker run -p 8501:8501 plant-disease-rag-assistant
```

This will launch the Streamlit UI at [http://localhost:8501](http://localhost:8501).

## Project Structure

```
plant-disease-rag-assistant/
│
├── README.md                  # Project overview, setup, usage, and documentation links
├── docker-compose.yml         # Multi-container orchestration for API and UI services
├── requirements.txt           # Python dependencies for the project
├── .env.example               # Example environment variables for local/dev setup
├── Makefile                   # Common build, run, and test commands
│
├── data/
│   ├── raw/                   # Unprocessed source datasets (images, docs, etc.)
│   ├── processed/             # Cleaned/normalized data ready for ingestion
│   ├── kb/                    # Final knowledge base files (JSON, Parquet, etc.)
│   ├── sample_kb/             # Tiny sample KB for quick demos and cloud deployment
│
├── notebooks/
│   ├── 01_classifier_review.ipynb   # EDA and review of image classifier results
│   ├── 02_doc_ingestion.ipynb       # KB ingestion and document processing experiments
│   └── 03_rag_experiments.ipynb     # RAG pipeline and retrieval experiments
│
├── src/
│   ├── classifier/            # Image classification models and utilities
│   ├── ingestion/             # Scripts for scraping, cleaning, and building the KB
│   │   ├── build_kb.py                # Main ingestion script: collects, chunks, normalizes, and deduplicates raw data from sources (PlantVillage, Wikipedia...) to build the KB.
│   │   ├── refresh_kb_descriptions.py # Enrichment script: updates or fills in missing fields in the KB by scraping PlantVillage pages.
│   │   ├── import_dataset_plants.py   # Utility for importing and summarizing external datasets
│   │   ├── validate_kb_urls.py        # Utility script for validating URLs in the KB.
│   │   ├── scrape_plantvillage_infos.py # Scraper for PlantVillage 'infos' pages, used for initial disease/crop info extraction.
│   │   └── ...                        # Other helpers/utilities for KB construction and cleaning.
│   ├── retrieval/             # BM25, FAISS, and hybrid retrieval logic
│   ├── llm/                   # RAG pipeline, LLM integration, and prompt logic
│   ├── monitoring/            # Logging, metrics, and health checks
│   └── interface/
│       ├── streamlit_app.py   # Main Streamlit UI for interactive Q&A and classification
│       ├── app_gradio.py      # (Obsolete) Gradio UI for quick demo/testing; use Streamlit for all new features
│       └── api.py             # FastAPI backend for RESTful endpoints
│
├── tests/
│   └── test_retrieval.py      # Unit tests for retrieval and RAG pipeline
│
└── docs/
    ├── architecture.png       # System architecture diagram
    ├── project_plan.md        # Project goals, milestones, and planning notes
    ├── STREAMLIT.md           # Streamlit UI usage and customization guide
    ├── data_card.md           # Dataset sources, build steps, and limitations
    ├── retrieval.md           # Retrieval pipeline documentation
    ├── testing.md             # Testing and debugging instructions
    ├── improvements.md        # Roadmap, backlog, and enhancement ideas
    └── artifacts.md           # Evaluation results and experiment artifacts
```

## API Endpoints

- **GET `/health`**  
  Returns API and pipeline status, configuration, and metadata (index path, device, top_k, document count, model info).

- **POST `/rag`**  
  Accepts a user query and optional filters (plant, disease, top_k, fusion, alpha, temperature, timeout).  
  Runs retrieval and LLM answer generation, returning the answer, sources, retrieved chunks, and latency.

- **POST `/api/classify`**  
  Accepts an image file upload (plant leaf).  
  Runs image classification using the ViT model and returns the top predicted disease labels and scores.

- **POST `/api/feedback`**  
  Accepts feedback about answers (thumbs up/down, comments, etc.) as JSON.  
  Stores feedback for monitoring and dashboarding.

Example payloads for each endpoint are available in the code comments.

## Usage

### Local Development

- Create environment and install dependencies:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

- Run Streamlit UI:
```powershell
streamlit run src/interface/streamlit_app.py
```

- Run FastAPI backend:
```powershell
uvicorn src.interface.api:app --host 0.0.0.0 --port 8000
```

- Run Gradio UI (deprecated, for intermediate development demos only):
```powershell
python src/interface/app_gradio.py
```

### Web UIs

- **Streamlit**: Main interface UI for image classification, RAG assisted search.
- See [docs/STREAMLIT.md](docs/STREAMLIT.md) for details.

## Deployment Options

This project delivers:
- **A ready-to-use Hugging Face Spaces app** powered by Streamlit for interactive plant disease diagnosis and Q&A.  
  [🌿 Try it online here.](https://huggingface.co/spaces/mcherif/Plant-Disease-RAG-Assistant)
- **A FastAPI backend** exposing RESTful endpoints for programmatic access, integration, and automation.

You can use the Streamlit app directly on Hugging Face, or deploy the FastAPI service for custom workflows and integrations.

## Dataset

See [docs/data_card.md](docs/data_card.md) for sources, build steps, and limitations.

### Expanding Disease Coverage

- Update seed pairs in [`data/plantvillage_kb.json`](data/plantvillage_kb.json) (via [`src/ingestion/refresh_kb_descriptions.py`](src/ingestion/refresh_kb_descriptions.py) or manual edit).
- Extend normalization and relevance filters in [`src/ingestion/build_kb.py`](src/ingestion/build_kb.py).
- Rebuild KB and re-index retrieval store:
```powershell
python -m src.ingestion.build_kb --sources plantvillage,wikipedia --out data/kb --min_tokens 50 --max_tokens 400 --overlap 80 --dedup minhash --dedup-threshold 0.9 --wiki-lang en --wiki-interval 0.5 --verbose
```

## Retrieval

See [docs/retrieval.md](docs/retrieval.md) for building FAISS index, BM25, and hybrid fusion. Evaluation artifacts in [docs/artifacts.md](docs/artifacts.md).

## Testing

See [docs/testing.md](docs/testing.md) for running tests, OpenAI integration, and debugging tips.

## Supported Plants and Diseases

This project supports image classification for a specific set of plants and their associated diseases. The full list is maintained in [`plant_diseases_table.csv`](plant_diseases_table.csv).

### Viewing the List

You can view the current list of supported plants and diseases by opening [`plant_diseases_table.csv`](plant_diseases_table.csv) in any spreadsheet or text editor. Each row corresponds to a plant, and the columns list the diseases that can be detected for that plant.

### Updating the List

**Do not edit the CSV manually.**  
To regenerate or update the list, run the following script:

```powershell
python scripts/list_all_plant_diseases_covered.py
```

This will rebuild `plant_diseases_table.csv` from the latest knowledge base and classifier configuration.

If you wish to expand disease coverage, update the relevant source files (e.g., `data/plantvillage_kb.json`) and then rerun the script above.

## Roadmap

See [docs/improvements.md](docs/improvements.md) for prioritized backlog and test ideas.

## Notes

- Place unversioned datasets/docs in `data/raw`, `data/processed`, `data/kb`.
- Add/update ingestion scripts under `src/ingestion`, retrieval logic under `src/retrieval`.
- Configure RAG pipeline in `src/llm/rag_pipeline.py`.

## License

MIT
