# Project Makefile: common dev tasks (Windows-friendly).
# Requires GNU Make (e.g., Chocolatey: choco install make -y).
# Use TABs (not spaces) to indent recipe lines.

.PHONY: install kb kb-all index eval_retrieval eval_rag rag_qa eval_rag_full ui api sample_index run_sample_ui run_sample_api docker_build docker_up docker_down docker_logs docker_api_shell

install:
	pip install -r requirements.txt

# Build a KB from PlantVillage into data/kb (chunks + manifest)
# Adjust token bounds/overlap as needed. Wikipedia ingestion is stubbed for now.
kb:
	python -m src.ingestion.build_kb --sources plantvillage --out data/kb --min_tokens 10 --max_tokens 1000 --overlap 100 --dedup minhash --dedup-threshold 0.9 --verbose

# Build PV + Wikipedia (polite)
kb-all:
	python -m src.ingestion.build_kb --sources plantvillage,wikipedia --out data/kb --min_tokens 10 --max_tokens 1000 --overlap 100 --dedup minhash --dedup-threshold 0.9 --wiki-lang en --wiki-interval 0.5 --verbose

# Build FAISS index from the KB manifest
index:
	python -m src.retrieval.build_index --manifest data\kb\manifest.parquet --out models\index\kb-faiss-bge --model BAAI/bge-small-en-v1.5 --batch-size 64 --device cpu --doc-prefix "passage: "

# Offline retrieval evaluation (Recall@k, nDCG@k)
eval_retrieval:
	python -m src.retrieval.evaluate --index-dir models\index\kb-faiss-bge --labels data\kb\labels.jsonl --device cuda --fusion sum --alpha 0.7 --top-ks 5,10

# RAG evaluation with LLM-as-judge (faithfulness/relevance)
# Requires OPENAI_API_KEY; with --skip-if-no-key it exits cleanly when unset.
# Artifacts: artifacts/rag_eval/rag_eval.json and rag_eval.csv
eval_rag:
	python -m src.eval.evaluate_rag --dataset data/kb/labels.jsonl --out artifacts/rag_eval --n 50 --skip-if-no-key

rag_qa:
	python -m src.eval.make_rag_qa

eval_rag_full:
	python -m src.eval.evaluate_rag --dataset data/eval/rag_qa.jsonl --out artifacts/rag_eval --n 120 --skip-if-no-key

ui:
	set PYTHONPATH=.&& streamlit run src\interface\streamlit_app.py

api:
	set PYTHONPATH=.&& uvicorn src.interface.api:app --reload --host 127.0.0.1 --port 8000

sample_index:
	python -m src.ingestion.build_sample_kb

run_sample_ui:
	set INDEX_DIR=models/index/sample-faiss-bge&& set PYTHONPATH=.&& streamlit run src\interface\streamlit_app.py

run_sample_api:
	set INDEX_DIR=models/index/sample-faiss-bge&& set PYTHONPATH=.&& uvicorn src.interface.api:app --host 127.0.0.1 --port 8000

docker_build:
	docker build -t plant-disease-rag:latest .

docker_up:
	docker compose up --build -d

docker_down:
	docker compose down

docker_logs:
	docker compose logs -f

docker_api_shell:
	docker exec -it plant_rag_api bash
