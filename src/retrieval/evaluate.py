"""
Evaluator for hybrid retrieval (FAISS + BM25), reporting Recall@k and nDCG@k.

It:
- Loads a FAISS index and aligned metadata produced by src/retrieval/build_index.py
- Reads a labels.jsonl file with queries and relevance hints
- Encodes queries with the same sentence-embedding model used at indexing
- Retrieves FAISS candidates, optionally fuses with BM25 over candidates (sum or RRF)
- Applies simple metadata filters (e.g., plant)
- Computes per-query and aggregate metrics (Recall@k, nDCG@k)
- Writes results to artifacts/retrieval_eval/retrieval_eval.json

Label schema (one JSON per line; any subset is okay):
  {
    "query": "tomato yellow leaf curl symptoms",
    "plant": "Tomato",
    "diseases": ["tomato yellow leaf curl virus", "tylcv"],   # substring match on disease/title
    "doc_ids": ["<optional explicit doc_id>"],                 # exact match if provided
    "title_contains": ["yellow leaf curl"]                     # substring match on title
  }

Example:
  python -m src.retrieval.evaluate ^
    --index-dir models\\index\\kb-faiss-bge ^
    --labels data\\kb\\labels.jsonl ^
    --device cuda --fusion sum --alpha 0.7 --top-ks 5,10
"""
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import faiss  # type: ignore
import numpy as np
from rank_bm25 import BM25Okapi
import warnings
from sentence_transformers import SentenceTransformer


def load_meta(index_dir: Path) -> List[Dict]:
    """Load metadata rows (one per vector) saved alongside the FAISS index."""
    meta_path = index_dir / "meta.jsonl"
    rows: List[Dict[str, Any]] = []
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    warnings.warn(
                        f"[eval] Could not parse {meta_path.name}; using fallback KB data.",
                        RuntimeWarning,
                    )
                    rows = _load_meta_fallback(index_dir)
                    break
    except FileNotFoundError:
        warnings.warn(
            f"[eval] Metadata file {meta_path} missing; using fallback KB data.",
            RuntimeWarning,
        )
        rows = _load_meta_fallback(index_dir)
    if not rows:
        warnings.warn(
            f"[eval] Metadata empty in {meta_path}; using fallback KB data.",
            RuntimeWarning,
        )
        rows = _load_meta_fallback(index_dir)
    return _hydrate_meta_rows(rows)

def load_config(index_dir: Path) -> Dict:
    """Load index configuration saved by build_index.py (config.json).

    Returns:
        Dict: configuration including model name, vector dim, doc_prefix, etc.
    """
    try:
        return json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    except Exception:
        return {
            "model": "BAAI/bge-small-en-v1.5",
            "doc_prefix": "passage: ",
            "normalize": True,
            "metric": "cosine (IP on normalized vectors)",
            "dim": 384,
        }


def _tok(s: str) -> List[str]:
    """Lightweight tokenizer for BM25: lowercase and extract word tokens.

    Args:
        s: input string.

    Returns:
        List[str]: alphanumeric tokens in lowercase.
    """
    return re.findall(r"\w+", str(s).lower())


def _embed_query(
    text: str,
    model_name: str,
    device: str = "cpu",
    encoder: SentenceTransformer | None = None,
) -> np.ndarray:
    """Encode a single query into a normalized embedding for cosine similarity.

    Args:
        text: query text (already prefixed if required by the model family).
        model_name: SentenceTransformers model or HF hub ID used at indexing.
        device: "cpu" or "cuda".
        encoder: optional pre-loaded SentenceTransformer instance to reuse.

    Returns:
        np.ndarray: shape (1, dim) float32, L2-normalized.
    """
    model = encoder or SentenceTransformer(model_name, device=device)
    v = model.encode([text], convert_to_numpy=True,
                     normalize_embeddings=False).astype("float32")
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v

def _build_fallback_index(meta: List[Dict], encoder: SentenceTransformer, doc_prefix: str) -> faiss.Index:
    """Rebuild a simple inner-product FAISS index when the persisted file is incompatible."""
    docs: List[str] = []
    prefix = doc_prefix or ""
    for row in meta:
        text = _ensure_text(row)
        docs.append(f"{prefix}{text}" if prefix else text)

    if not docs:
        dim = encoder.get_sentence_embedding_dimension()
        return faiss.IndexFlatIP(dim)

    embeddings = encoder.encode(
        docs,
        batch_size=min(64, len(docs)),
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")
    embeddings /= (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def _load_meta_fallback(index_dir: Path) -> List[Dict]:
    kb_path = _locate_fallback_kb(index_dir)
    try:
        raw = json.loads(kb_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        warnings.warn(
            f"[eval] Fallback KB not available at {kb_path}; using bundled sample KB.",
            RuntimeWarning,
        )
        kb_path = _locate_fallback_kb(index_dir, sample_only=True)
        raw = json.loads(kb_path.read_text(encoding="utf-8"))
    return [dict(row) for row in raw if isinstance(row, dict)]


def _locate_fallback_kb(index_dir: Path, sample_only: bool = False) -> Path:
    for base in [index_dir, *index_dir.parents]:
        candidate = base / "data" / "kb" / ("sample_kb.json" if sample_only else "kb.json")
        if candidate.exists():
            return candidate
    return Path("data/sample_kb/sample_kb.json") if sample_only else Path("data/kb/sample_kb.json")


def _hydrate_meta_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hydrated: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        for key in ("title", "plant", "disease", "url", "doc_id"):
            if key in item and item[key] is not None:
                item[key] = str(item[key])
        _ensure_text(item)
        hydrated.append(item)
    if not hydrated:
        raise RuntimeError("No metadata rows available for retrieval.")
    return hydrated


def _ensure_text(row: Dict[str, Any]) -> str:
    text = str(row.get("text") or "").strip()
    if not text:
        parts = []
        for key in ("description", "symptoms", "cause", "management", "prevention"):
            val = row.get(key)
            if val:
                parts.append(str(val))
        if not parts:
            for key in ("title", "plant", "disease"):
                val = row.get(key)
                if val:
                    parts.append(str(val))
        text = "\n".join(part.strip() for part in parts if part).strip()
    row["text"] = text
    return text

def _infer_query_prefix(cfg: Dict, given: str) -> str:
    """Determine query prefix if not explicitly provided.

    If docs were indexed with 'passage: ' (e.g., BGE/E5/GTE), return 'query: '.
    Otherwise, return empty string.

    Args:
        cfg: index config dict (may contain 'doc_prefix').
        given: user-specified query prefix (wins if non-empty).

    Returns:
        str: inferred or provided query prefix.
    """
    if given:
        return given
    doc_prefix = str(cfg.get("doc_prefix", "")).strip().lower()
    return "query: " if doc_prefix.startswith("passage:") else ""


def _relevant_set(meta: List[Dict], label: Dict) -> Set[int]:
    """Build the set of relevant vector indices for a given label.

    Priority:
      1) Explicit doc_ids (exact matches)
      2) Substring matches on disease or title (case-insensitive)
    Filtered by plant if provided in the label.

    Args:
        meta: metadata list aligned with vectors.
        label: one label line from labels.jsonl.

    Returns:
        Set[int]: indices into meta considered relevant for this query.
    """
    plant = str(label.get("plant", "")).lower().strip()
    rel: Set[int] = set()

    # Highest fidelity: explicit doc_ids
    doc_ids = set(label.get("doc_ids", []) or [])
    if doc_ids:
        for i, m in enumerate(meta):
            if plant and str(m.get("plant", "")).lower() != plant:
                continue
            if str(m.get("doc_id", "")) in doc_ids:
                rel.add(i)

    # Match by disease/title substrings (case-insensitive)
    diseases = [d.lower() for d in (label.get("diseases", []) or [])]
    title_contains = [t.lower()
                      for t in (label.get("title_contains", []) or [])]
    for i, m in enumerate(meta):
        if plant and str(m.get("plant", "")).lower() != plant:
            continue
        d = str(m.get("disease", "")).lower()
        t = str(m.get("title", "")).lower()
        if any(x and x in d for x in diseases) or any(x and x in t for x in title_contains):
            rel.add(i)

    return rel


def _minmax(x: np.ndarray) -> np.ndarray:
    """Min-max normalize a 1D score array to [0, 1] (per query).

    Args:
        x: array of scores.

    Returns:
        np.ndarray: normalized scores; zeros if all values are equal.
    """
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn + 1e-12) if mx > mn else np.zeros_like(x)


def _metrics_at(ranked: List[int], relevant: Set[int], k: int) -> Tuple[float, float]:
    """Compute Recall@k and nDCG@k (binary relevance).

    Args:
        ranked: full ranked list of candidate indices (post-filtering).
        relevant: set of indices considered relevant for this query.
        k: cutoff.

    Returns:
        (recall, ndcg): both in [0, 1]. nDCG@k=1 means ideal ordering of relevant items in top-k.
    """
    top = ranked[:k]
    # Recall@k
    hits = sum(1 for i in top if i in relevant)
    recall = hits / max(1, len(relevant))
    # nDCG@k (binary gains)
    dcg = 0.0
    for r, i in enumerate(top, start=1):
        gain = 1.0 if i in relevant else 0.0
        dcg += gain / np.log2(r + 1)
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / np.log2(r + 1)
               for r in range(1, ideal_hits + 1)) if ideal_hits > 0 else 0.0
    ndcg = (dcg / idcg) if idcg > 0 else 0.0
    return recall, ndcg


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the evaluator.

    Important flags:
      --index-dir     Path to FAISS index directory (with faiss.index, meta.jsonl, config.json)
      --labels        Path to labels.jsonl (queries + relevance hints)
      --fusion        none | sum | rrf (late fusion method)
      --alpha         Weight for vector score in weighted-sum fusion
      --pretopk       Number of FAISS candidates before fusion/filtering
      --top-ks        Comma-separated list of cutoffs (e.g., 5,10)
      --query-prefix  Override inferred query prefix (e.g., 'query: ')
      --device        cpu | cuda
    """
    ap = argparse.ArgumentParser(
        description="Evaluate hybrid retrieval (Recall@k, nDCG@k)")
    ap.add_argument("--index-dir", default="models/index/kb-faiss-bge")
    ap.add_argument("--labels", default="data/kb/labels.jsonl")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--model", default=None,
                    help="Override model (defaults to index config)")
    ap.add_argument("--fusion", default="sum", choices=["none", "sum", "rrf"])
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="Weight for vector score in 'sum'")
    ap.add_argument("--pretopk", type=int, default=100,
                    help="FAISS candidates before fusion")
    ap.add_argument("--top-ks", default="5,10", help="Comma list of k values")
    ap.add_argument("--query-prefix", default="",
                    help="Prefix for queries (e.g., 'query: ')")
    ap.add_argument("--out", default="artifacts/retrieval_eval")
    return ap.parse_args()


def main() -> int:
    """Run evaluation over all label queries and write aggregate metrics.

    Returns:
        int: process exit code (0 on success).
    """
    args = parse_args()
    index_dir = Path(args.index_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(index_dir)
    meta = load_meta(index_dir)
    model_name = args.model or cfg.get(
        "model", "sentence-transformers/all-MiniLM-L6-v2")
    encoder_model = SentenceTransformer(model_name, device=args.device)

    allow_fallback = os.getenv("RAG_ALLOW_FAISS_FALLBACK", "1").lower() not in {"0", "false"}
    try:
        index = faiss.read_index(str(index_dir / "faiss.index"))
    except RuntimeError as err:
        if not allow_fallback:
            raise
        warnings.warn(
            f"[eval] Failed to load FAISS index '{index_dir / 'faiss.index'}'; rebuilding at runtime: {err}",
            RuntimeWarning,
        )
        index = _build_fallback_index(meta, encoder_model, str(cfg.get("doc_prefix", "")))

    if index.ntotal != len(meta):
        raise RuntimeError(
            f"Index size {index.ntotal} != meta rows {len(meta)}")

    q_prefix = _infer_query_prefix(cfg, args.query_prefix)
    top_ks = [int(x) for x in args.top_ks.split(",") if x.strip()]

    # Precompute BM25 tokens for candidate reranking when needed
    # (we build per-query on candidate texts only)
    results = []
    with Path(args.labels).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            lbl = json.loads(line)
            query = str(lbl["query"])
            plant = str(lbl.get("plant", "")).strip() or None

            q = _embed_query(f"{q_prefix}{query}" if q_prefix else query,
                             model_name=model_name, device=args.device, encoder=encoder_model)
            vec_scores, vec_idxs = index.search(
                q, max(args.pretopk, max(top_ks)))
            vec_scores, vec_idxs = vec_scores[0], vec_idxs[0]

            # Candidate pool
            cand = list(zip(vec_scores.tolist(), vec_idxs.tolist()))

            # Optional BM25 fusion over candidates
            if args.fusion != "none" and cand:
                cand_idxs = [i for _, i in cand]
                cand_texts = [meta[i].get("text", "") for i in cand_idxs]
                bm25 = BM25Okapi([_tok(t) for t in cand_texts])
                bm_scores = np.array(bm25.get_scores(
                    _tok(query)), dtype=np.float32)

                v = np.array([s for s, _ in cand], dtype=np.float32)
                v_n = _minmax(v)
                b_n = _minmax(bm_scores)
                if args.fusion == "sum":
                    fused = args.alpha * v_n + (1.0 - args.alpha) * b_n
                    order = np.argsort(fused)[::-1].tolist()
                    cand = [(float(v[j]), cand_idxs[j]) for j in order]
                else:  # rrf
                    k_rrf = 60.0
                    vec_rank = {i: r for r, (_, i) in enumerate(
                        sorted(cand, key=lambda x: x[0], reverse=True), 1)}
                    bm_order = np.argsort(bm_scores)[::-1].tolist()
                    bm_rank = {cand_idxs[j]: r for r,
                               j in enumerate(bm_order, 1)}
                    scored = []
                    for i in cand_idxs:
                        s = 1.0 / (k_rrf + vec_rank[i]) + \
                            1.0 / (k_rrf + bm_rank[i])
                        scored.append((s, i))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    vmap = {i: s for s, i in cand}
                    cand = [(float(vmap[i]), i) for _, i in scored]

            # Apply plant filter like the CLI
            ranked_idxs = []
            for s, i in cand:
                m = meta[i]
                if plant and str(m.get("plant", "")).lower() != plant.lower():
                    continue
                ranked_idxs.append(i)

            relevant = _relevant_set(meta, lbl)
            per_k = {}
            for k in top_ks:
                rec, ndcg = _metrics_at(ranked_idxs, relevant, k)
                per_k[str(k)] = {"recall": rec, "ndcg": ndcg}

            results.append({"query": query, "plant": plant, "metrics": per_k})

    # Aggregate
    agg = {str(k): {"recall": 0.0, "ndcg": 0.0} for k in top_ks}
    for r in results:
        for k, m in r["metrics"].items():
            agg[k]["recall"] += m["recall"]
            agg[k]["ndcg"] += m["ndcg"]
    n = max(1, len(results))
    for k in agg:
        agg[k]["recall"] /= n
        agg[k]["ndcg"] /= n

    # Save
    (out_dir / "retrieval_eval.json").write_text(json.dumps(
        {"aggregate": agg, "per_query": results}, indent=2), encoding="utf-8")
    print("[eval] aggregate:", json.dumps(agg, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())