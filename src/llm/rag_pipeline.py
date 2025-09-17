"""
RAG pipeline: retrieve → compose prompt with citations → call LLM.

Overview
- Retrieval: FAISS dense vector search over KB chunk embeddings, with optional BM25
  late-fusion (weighted sum or Reciprocal Rank Fusion) to blend lexical + semantic signals.
- Prompting: Compose a concise prompt containing the user's question and a numbered
  context block. The LLM is instructed to cite sources inline as [n] matching the
  provided context entries.
- Generation: Call an LLM backend (OpenAI by default) and return an answer with
  citations plus the source metadata.

Index layout (produced by src/retrieval/build_index.py under models/index/)
- <index_dir>/faiss.index        — FAISS index with vectors
- <index_dir>/meta.jsonl         — one JSON per vector (aligned order) with fields like:
                                   { "doc_id", "title", "plant", "disease", "url", "text", ... }
- <index_dir>/config.json        — includes model name, dim, and optional "doc_prefix"

Model prefixes
- Some embedding families (BGE, E5, GTE) benefit from special prefixes:
  - Documents indexed with "passage: " (doc_prefix)
  - Queries encoded with "query: " (inferred here automatically)
- If no doc_prefix is present, queries are used as-is.

Fusion methods
- none: pure vector similarity from FAISS.
- sum: per-query min–max normalization then fused = alpha * vec + (1 - alpha) * bm25.
- rrf: Reciprocal Rank Fusion combining ranks from vector and BM25 lists.

Defaults and how to change
- Default fusion is "sum" with alpha=0.7 (see RetrievalConfig defaults).
- Programmatic control:
  - Global: cfg = RetrievalConfig(..., fusion="rrf", alpha=0.6); rag = RAGPipeline(cfg)
  - Per-call: rag.answer("...", fusion="none"); rag.answer("...", fusion="sum", alpha=0.5)
- Evaluator CLI examples:
  - python -m src/retrieval.evaluate --fusion sum --alpha 0.7
  - python -m src/retrieval.evaluate --fusion rrf
  - python -m src/retrieval.evaluate --fusion none

Environment
- OPENAI_API_KEY must be set to use the default OpenAI backend; OPENAI_MODEL overrides
  model name (default: gpt-4o-mini).

Usage example
    from pathlib import Path
    from src.llm.rag_pipeline import RAGPipeline, RetrievalConfig

    cfg = RetrievalConfig(
        index_dir=Path("models/index/kb-faiss-bge"),
        device="cuda",
        fusion="sum",
        alpha=0.7,
        top_k=4,
    )
    rag = RAGPipeline(cfg)
    res = rag.answer("tomato yellow leaf curl symptoms", plant="Tomato")
    print(res["answer"])
    print("Sources:", [(s["id"], s["title"], s["url"]) for s in res["sources"]])

Returns schema (answer())
{
  "answer": str,                         # model output with inline [n] citations
  "sources": [ { "id": int, "title": str, "url": str }, ... ],   # numbered to match citations
  "retrieved": [ { "score": float, "meta": { ... } }, ... ]      # raw top-k retrieval results
}

Notes
- Performance: The encoder is instantiated once and cached in the pipeline instance.
- Portability: _load_prompt expects src/llm/prompts/answer.txt. Keep that template synced.
- Extensibility: Swap _generate() with a different backend, add rerankers, or
  implement safety filters before returning the final answer.
"""
from __future__ import annotations

import json
import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import faiss  # type: ignore
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import unicodedata
import difflib

def _tok(s: str) -> List[str]:
    """Lightweight tokenizer for BM25: lowercase alphanumeric tokens."""
    return re.findall(r"\w+", str(s).lower())

def _minmax(x: np.ndarray) -> np.ndarray:
    """Min–max normalize a 1D score array to [0, 1]; returns zeros if degenerate."""
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn + 1e-12) if mx > mn else np.zeros_like(x)

def _norm(v: Any) -> str:
    return str(v or "").strip().lower()

def normalize_for_index(text):
    # Lowercase, remove accents, strip punctuation, etc.
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def _is_match(query_val, meta_val, threshold=0.7):
    """Return True if query_val and meta_val are a close match (substring or fuzzy)."""
    if not query_val or not meta_val:
        return True  # treat empty as wildcard
    # Substring match (either direction)
    if query_val in meta_val or meta_val in query_val:
        return True
    # Fuzzy match
    ratio = difflib.SequenceMatcher(None, query_val, meta_val).ratio()
    return ratio >= threshold

@dataclass
class RetrievalConfig:
    """Configuration for the retrieval phase.

    Args:
        index_dir: Directory containing faiss.index, meta.jsonl, and config.json.
        device: "cpu" or "cuda" for query encoding.
        model_name: Optional override for the embedding model (defaults to index config).
        fusion: "none" | "sum" | "rrf" — how to combine vector and BM25 signals.
        alpha: Weight for vector score in "sum" fusion (0..1). Higher favors embeddings.
        top_k: Number of chunks to return to the LLM context.
    """
    index_dir: Path
    device: str = "cpu"
    model_name: Optional[str] = None
    fusion: str = "sum"  # none | sum | rrf
    alpha: float = 0.7
    top_k: int = 4
    # refuse if best retrieval score < min_score
    min_score: float | None = None
    refusal_message: str = (
        "Sorry, I don't have enough context to answer confidently. "
        "Try rephrasing the question or narrowing by plant/disease."
    )

class RAGPipeline:
    """RAG pipeline orchestrating retrieval, prompt composition, and generation.

    Responsibilities:
    - Load FAISS index + aligned metadata and index config.
    - Encode queries (adding "query: " if documents were indexed with "passage: ").
    - Retrieve candidates via FAISS and optionally fuse with BM25 over candidates.
    - Compose a prompt with numbered context blocks to enable inline citations.
    - Call an LLM backend to produce the final answer.

    Raises:
        RuntimeError: if the index files are inconsistent or the LLM backend fails.
    """

    def __init__(self, cfg: RetrievalConfig):
        """Initialize the pipeline and cache encoder and index handles."""
        self.cfg = cfg
        self.index_dir = Path(cfg.index_dir)

        # Locate FAISS index file (support multiple possible filenames)
        primary = self.index_dir / "faiss.index"
        if not primary.exists():
            candidates = []
            for pat in ("*.index", "*.faiss", "index.faiss"):
                candidates.extend(self.index_dir.glob(pat))
            if candidates:
                primary = candidates[0]
        if not primary.exists():
            raise FileNotFoundError(
                f"FAISS index file not found in {self.index_dir}")

        # Load metadata + config before encoder/index init so we can rebuild if needed
        self.meta = self._load_meta(self.index_dir)
        self.config = self._load_config(self.index_dir)

        self.model_name = cfg.model_name or self.config.get(
            "model", "sentence-transformers/all-MiniLM-L6-v2")
        self.doc_prefix = str(self.config.get("doc_prefix", ""))
        doc_prefix_norm = self.doc_prefix.strip().lower()
        self.query_prefix = "query: " if doc_prefix_norm.startswith(
            "passage:") else ""
        self.encoder = SentenceTransformer(self.model_name, device=cfg.device)

        self._faiss_fallback = False
        self._faiss_fallback_reason: Optional[str] = None
        allow_fallback = os.getenv("RAG_ALLOW_FAISS_FALLBACK", "1").lower() not in {"0", "false"}
        try:
            self.index = faiss.read_index(str(primary))
        except RuntimeError as err:
            if not allow_fallback:
                raise
            warnings.warn(
                f"[rag] Failed to load FAISS index '{primary.name}', rebuilding an in-memory index: {err}",
                RuntimeWarning,
            )
            self.index = self._build_fallback_index()
            self._faiss_fallback = True
            self._faiss_fallback_reason = str(err)

        # Sanity check: vector count must match metadata rows
        if self.index.ntotal != len(self.meta):
            raise RuntimeError(
                f"Index size {self.index.ntotal} != meta rows {len(self.meta)}")

    def _load_meta(self, index_dir: Path) -> List[Dict]:
        """Load metadata rows, falling back to a local KB snapshot when necessary."""
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
                            f"[rag] Could not parse {meta_path.name}; using fallback KB data.",
                            RuntimeWarning,
                        )
                        rows = self._load_meta_fallback()
                        break
        except FileNotFoundError:
            warnings.warn(
                f"[rag] Metadata file {meta_path} missing; using fallback KB data.",
                RuntimeWarning,
            )
            rows = self._load_meta_fallback()
        if not rows:
            warnings.warn(
                f"[rag] Metadata empty in {meta_path}; using fallback KB data.",
                RuntimeWarning,
            )
            rows = self._load_meta_fallback()
        return self._hydrate_meta_rows(rows)

    @staticmethod
    def _load_config(index_dir: Path) -> Dict:
        """Load index configuration (e.g., model name, dim, doc_prefix)."""
        return json.loads((index_dir / "config.json").read_text(encoding="utf-8"))

    def _load_meta_fallback(self) -> List[Dict]:
        kb_path = self._locate_fallback_kb()
        try:
            raw = json.loads(kb_path.read_text(encoding="utf-8"))
        except Exception as err:  # noqa: BLE001
            warnings.warn(
                f"[rag] Fallback KB not available at {kb_path}; using bundled sample KB.",
                RuntimeWarning,
            )
            kb_path = self._locate_fallback_kb(sample_only=True)
            raw = json.loads(kb_path.read_text(encoding="utf-8"))
        return [dict(row) for row in raw if isinstance(row, dict)]

    def _locate_fallback_kb(self, sample_only: bool = False) -> Path:
        search_roots = [self.index_dir, *self.index_dir.parents]
        for base in search_roots:
            candidate = base / "data" / "kb" / ("sample_kb.json" if sample_only else "kb.json")
            if candidate.exists():
                return candidate
        # Last resort: project-relative lookup
        return Path("data/sample_kb/sample_kb.json") if sample_only else Path("data/kb/sample_kb.json")

    def _hydrate_meta_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        hydrated: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            for key in ("title", "plant", "disease", "url", "doc_id"):
                if key in item and item[key] is not None:
                    item[key] = str(item[key])
            self._ensure_text(item)
            hydrated.append(item)
        if not hydrated:
            raise RuntimeError("No metadata rows available for retrieval.")
        return hydrated

    def _ensure_text(self, row: Dict[str, Any]) -> str:
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

    def _build_fallback_index(self) -> faiss.Index:
        """Rebuild a simple inner-product index when the persisted FAISS index is incompatible."""
        docs: List[str] = []
        prefix = self.doc_prefix or ""
        for row in self.meta:
            text = self._ensure_text(row)
            docs.append(f"{prefix}{text}" if prefix else text)

        if not docs:
            dim = self.encoder.get_sentence_embedding_dimension()
            return faiss.IndexFlatIP(dim)

        embeddings = self.encoder.encode(
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

    def _embed_query(self, text: str) -> np.ndarray:
        """Encode a single query; returns L2-normalized float32 vector of shape (1, dim)."""
        v = self.encoder.encode(
            [text], convert_to_numpy=True, normalize_embeddings=False).astype("float32")
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        return v

    def _retrieve(
        self,
        query: str,
        plant: Optional[str] = None,
        disease: Optional[str] = None,
        top_k: Optional[int] = None,
        fusion: Optional[str] = None,
        alpha: Optional[float] = None,
    ) -> List[Tuple[float, int]]:
        """Return a list of (score, idx) hits into self.meta."""
        # Respect explicit 0 (useful for tests) and only fall back when None
        k = self.cfg.top_k if top_k is None else int(top_k)
        if k <= 0:
            return []
        fusion = fusion or self.cfg.fusion
        alpha = alpha if alpha is not None else self.cfg.alpha

        # Encode query (add "query: " if documents used "passage: ")
        q_text = f"{self.query_prefix}{query}" if self.query_prefix else query
        q = self._embed_query(q_text)

        # Initial candidate set from FAISS
        pretopk = max(50, k)
        vec_scores, vec_idxs = self.index.search(q, pretopk)
        cand = list(zip(vec_scores[0].tolist(), vec_idxs[0].tolist()))

        # Optional hybrid fusion with BM25 over candidate texts
        if fusion != "none" and cand:
            cand_idxs = [i for _, i in cand]
            cand_texts = [self.meta[i].get("text", "") for i in cand_idxs]
            bm25 = BM25Okapi([_tok(t) for t in cand_texts])
            bm_scores = np.array(bm25.get_scores(
                _tok(query)), dtype=np.float32)

            v = np.array([s for s, _ in cand], dtype=np.float32)
            v_n = _minmax(v)
            b_n = _minmax(bm_scores)

            if fusion == "sum":
                # Weighted sum on normalized scores, then re-order candidates
                fused = alpha * v_n + (1.0 - alpha) * b_n
                order = np.argsort(fused)[::-1].tolist()
                cand = [(float(v[j]), cand_idxs[j]) for j in order]
            else:
                # Reciprocal Rank Fusion (RRF), robust to score-scale differences
                k_rrf = 60.0
                vec_rank = {i: r for r, (_, i) in enumerate(
                    sorted(cand, key=lambda x: x[0], reverse=True), 1)}
                bm_order = np.argsort(bm_scores)[::-1].tolist()
                bm_rank = {cand_idxs[j]: r for r, j in enumerate(bm_order, 1)}
                scored = []
                for i in cand_idxs:
                    s = 1.0 / (k_rrf + vec_rank[i]) + \
                        1.0 / (k_rrf + bm_rank[i])
                    scored.append((s, i))
                scored.sort(key=lambda x: x[0], reverse=True)
                # keep original vector scores for output
                vmap = {i: s for s, i in cand}
                cand = [(float(vmap[i]), i) for _, i in scored]

        # Apply metadata filters and truncate to top_k
        results: List[Tuple[float, int]] = []
        for s, idx in cand:  # cand must be (score, idx)
            meta = self.meta[idx]
            # Normalize both sides for robust matching
            meta_plant = normalize_for_index(meta.get("plant", ""))
            query_plant = normalize_for_index(plant) if plant else ""
            meta_disease = normalize_for_index(meta.get("disease", ""))
            query_disease = normalize_for_index(disease) if disease else ""
            # Use partial/fuzzy match
            if plant and not _is_match(query_plant, meta_plant):
                continue
            if disease and not _is_match(query_disease, meta_disease):
                continue
            results.append((float(s), int(idx)))
            if len(results) >= k:
                break
        return results

    def _load_prompt(self) -> str:
        """Load the answer prompt template.

        Expects an answer.txt template at src/llm/prompts/answer.txt within the repo.
        The template must include placeholders:
            {context} — concatenated, numbered source blocks
            {question} — the user question
        """
        p = self.index_dir.parents[2] / "src" / \
            "llm" / "prompts" / "answer.txt"
        return p.read_text(encoding="utf-8")

    def _compose(self, query: str, hits: List[Tuple[float, int]]) -> Tuple[str, List[Dict]]:
        """Compose the final prompt and assemble a compact sources list.

        Args:
            query: User question.
            hits: List of (score, meta_index) from retrieval.

        Returns:
            (prompt_text, sources):
              prompt_text: string with numbered context blocks and the question.
              sources: [{ "id": n, "title": str, "url": str }, ...] matching [n] citations.

        Note:
            - Context blocks are numbered in the same order as provided in hits.
            - Keep context compact to reduce token cost and improve focus.
        """
        sources = []
        context_blocks = []
        for rank, (_, idx) in enumerate(hits, 1):
            m = self.meta[idx]
            title = m.get("title", "") or m.get("disease", "") or "Source"
            url = m.get("url", "")
            text = m.get("text", "")
            sources.append({"id": rank, "title": title, "url": url})
            context_blocks.append(
                f"[{rank}] {title}\nURL: {url}\n---\n{text}\n")

        prompt_tpl = self._load_prompt()
        prompt = prompt_tpl.format(
            question=query, context="\n\n".join(context_blocks))
        return prompt, sources

    def _enforce_citations(self, text: str, sources: List[Dict]) -> str:
        """Ensure the answer contains at least one valid [n] citation within sources range.

        - If sources is empty, return as-is.
        - If no valid [n] appears, append " [1]" (maps to sources[0]['id'] which is 1).
        """
        if not sources:
            return text
        valid_ids = {s["id"] for s in sources}
        nums = [int(n) for n in re.findall(r"\[(\d+)\]", text)]
        has_valid = any(n in valid_ids for n in nums)
        if not has_valid:
            return (text.rstrip() + f" [{sources[0]['id']}]").strip()
        return text

    def _generate(self, prompt: str, model: Optional[str] = None, temperature: float = 0.1, timeout: float | None = None) -> str:
        """Call the LLM backend (OpenAI) and return the answer text.

        Args:
            prompt: Fully composed prompt including context and question.
            model: Optional override for the OpenAI model (OPENAI_MODEL default applies).

        Returns:
            LLM response text (str). May include inline [n] citations.

        Raises:
            RuntimeError: if OPENAI_API_KEY is missing or the API call fails.

        Tip:
            - Set environment variables:
                OPENAI_API_KEY=...    required
                OPENAI_MODEL=gpt-4o-mini (default) or another compatible model
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Set it to use the OpenAI backend.")
        try:
            # OpenAI Python SDK v1.x
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            system_content = (
                "You are a concise assistant for gardeners and farmers. "
                "Use simple, actionable language and practical steps to assist them with identifying plant diseases, best treatment and healthy practices. "
                "Cite sources inline as [n]. If context is insufficient, say you don't know. "
                "If recommending chemicals, remind to follow local regulations and label directions."
            )
            max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
            backoff = 1.0
            last_err = None
            for attempt in range(1, max_retries + 1):
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=temperature,
                        timeout=timeout,
                    )
                    return resp.choices[0].message.content or ""
                except Exception as e:
                    last_err = e
                    if attempt == max_retries:
                        break
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, 8.0)
            raise RuntimeError(
                f"LLM call failed after {max_retries} attempts: {last_err}")
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}") from e

    def answer(
        self,
        query: str,
        plant: Optional[str] = None,
        disease: Optional[str] = None,
        top_k: Optional[int] = None,
        fusion: Optional[str] = None,
        alpha: Optional[float] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run the full RAG flow and return answer plus sources and raw retrieval.

        Args:
            query: User question to be answered.
            plant: Optional plant filter for retrieval.
            disease: Optional disease filter for retrieval.
            top_k: Override for number of chunks used as context.
            fusion: Fusion strategy override ("none" | "sum" | "rrf").
            alpha: Weight override for "sum" fusion.
            model: LLM model override for the generation step.

        Returns:
            Dict with:
              - "answer": model output text (with inline [n] citations)
              - "sources": list of numbered source dicts (id, title, url)
              - "retrieved": list of {score, meta} for the returned top_k chunks

        Contract:
            - sources[n-1]["id"] equals citation index [n] used in the answer.
            - retrieved[i]["meta"] is an element from meta.jsonl (unchanged).
        """
        # Retrieve
        k = top_k or self.cfg.top_k
        hits: List[Tuple[float, int]] = self._retrieve(
            query=query, plant=plant, disease=disease, top_k=k, fusion=fusion, alpha=alpha
        )

        # Deduplicate indices (keep first/best score occurrence)
        dedup = []
        seen = set()
        for score, idx in hits:
            if idx in seen:
                continue
            dedup.append((score, idx))
            seen.add(idx)
        hits = dedup

        # Secondary de-dup: collapse multiple hits pointing to the same doc_id (defensive)
        doc_seen = set()
        uniq = []
        for score, idx in hits:
            doc_id = self.meta[idx].get("doc_id", f"idx-{idx}")
            if doc_id in doc_seen:
                continue
            doc_seen.add(doc_id)
            uniq.append((score, idx))
        hits = uniq

        # Guardrail: if no context (or explicitly requested top_k=0), refuse without calling LLM
        if (k <= 0) or (not hits):
            refusal = (
                "I don’t have enough relevant context to answer this. "
                "Please provide more details or try a different question."
            )
            return {"answer": refusal, "sources": [], "retrieved": []}

        # Build prompt with numbered contexts
        contexts: List[str] = [self.meta[idx].get(
            "text", "") for _, idx in hits]
        context_block = "\n\n".join(
            f"[{i+1}] {c}" for i, c in enumerate(contexts))
        header_bits = []
        if plant:
            header_bits.append(f"Plant: {plant}")
        if disease:
            header_bits.append(f"Disease: {disease}")
        header = " | ".join(header_bits) or "No labels"
        prompt = (
            "You are a plant pathology assistant. Use ONLY the provided context.\n"
            f"{header}\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context_block}\n\n"
            "Instructions:\n"
            "- Answer strictly about the plant/disease above. If not covered by context, say you don’t know.\n"
            "- Do not introduce other crops/diseases.\n"
            "- Cite sources as [n] matching the numbered context snippets.\n"
        )
        # Call LLM, compatible with mocked _generate that may not accept kwargs
        try:
            answer = self._generate(prompt, model=model,
                                    temperature=temperature, timeout=timeout)
        except TypeError:
            answer = self._generate(prompt, model=model)

        # Enforce at least one citation when we have sources but model returned none
        if hits and not re.search(r"\[\d+\]", answer or ""):
            answer = (answer or "").rstrip() + " [1]"

        # Shape outputs for UI
        retrieved = [{"score": float(s), "meta": self.meta[idx]}
                     for s, idx in hits]
        sources = [{
            "id": i + 1,
            "title": self.meta[idx].get("title") or self.meta[idx].get("disease") or "Source",
            "url": self.meta[idx].get("url", ""),
        } for i, (_, idx) in enumerate(hits)]

        return {"answer": answer, "retrieved": retrieved, "sources": sources}
