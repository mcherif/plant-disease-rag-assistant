"""Smoke test for the hybrid retrieval pipeline (FAISS or fallback embeddings).

Checks that a tomato TYLCV-style query surfaces expected plant/disease signals in the top results.
The test is skipped automatically when the index directory is missing (CI safety).
"""
# Tests for retrieval pipeline

from pathlib import Path

import pytest
from src.llm.rag_pipeline import RAGPipeline, RetrievalConfig

# Ensure the test is skipped if sentence_transformers isn't available
pytest.importorskip("sentence_transformers")


@pytest.mark.skipif(not Path("models/index/kb-faiss-bge").exists(), reason="Hybrid index not built")
def test_tylcv_retrieval_top5():
    """Top-5 retrieval contains Tomato + Yellow Leaf Curl signals (sanity check)."""
    index_dir = Path("models/index/kb-faiss-bge")

    cfg = RetrievalConfig(index_dir=index_dir, device="cpu", top_k=5)
    rag = RAGPipeline(cfg)
    query = "tomato yellow leaf curl symptoms"
    hits = rag._retrieve(query, plant="Tomato", top_k=5)
    top = [rag.meta[idx] for _, idx in hits]

    assert top, "no retrieval results returned"
    assert any("tomato" == str(r.get("plant", "")).lower() for r in top)
    assert any("yellow leaf curl" in str(r.get("disease", "")).lower() for r in top)
