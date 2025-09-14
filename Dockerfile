# syntax=docker/dockerfile:1

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*

# Copy only requirements first
COPY requirements.txt ./

# Copy wheels first to avoid repeated downloads
COPY wheels ./wheels

# Uses local wheels for faster builds and avoids repeated downloads of large packages.
# Install from local wheels first, then fallback to PyPI
RUN pip install --upgrade pip
RUN pip install --no-index --find-links=./wheels -r requirements.txt

# Now copy the rest of the project. This way, the pip install layer is only rebuilt if requirements.txt changes, not if any other file changes.
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN mkdir -p /root/.cache/pip
ENV PIP_CACHE_DIR=/root/.cache/pip

# Install deps (pyproject preferred, else requirements.txt)
RUN bash -c 'if [ -f pyproject.toml ]; then pip install --upgrade pip && pip install .; \
    elif [ -f requirements.txt ]; then pip install -r requirements.txt; \
    else echo "WARNING: no pyproject.toml or requirements.txt found"; fi'

# Build tiny sample index (idempotent)
RUN python -m src.ingestion.build_kb

ENV INDEX_DIR=models/index/kb-faiss-bge \
    RETRIEVAL_DEVICE=gpu \
    PORT_API=8000 \
    PORT_UI=8501

# Default = API (compose overrides for UI)
CMD ["streamlit", "run", "src/interface/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
CMD ["bash","-c","uvicorn src.interface.api:app --host 0.0.0.0 --port ${PORT_API}"]