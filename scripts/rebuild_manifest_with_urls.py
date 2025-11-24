"""
Rebuild data/kb/manifest.parquet from chunk files and kb.json with URLs/titles.

Why: The current manifest (and meta.jsonl) lost source URLs, so UI sources
can't render hyperlinks. This script reloads metadata from data/kb/kb.json,
aligns by doc_id, and writes a fresh manifest with url/title restored.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


KB_PATH = Path("data/kb/kb.json")
CHUNKS_DIR = Path("data/kb/chunks")
OUT_PATH = Path("data/kb/manifest.parquet")


def load_kb() -> Dict[str, dict]:
    kb = json.loads(Path(KB_PATH).read_text(encoding="utf-8"))
    return {row.get("doc_id"): row for row in kb}


def build_manifest_rows(kb_by_id: Dict[str, dict]) -> List[dict]:
    rows: List[dict] = []
    for chunk_path in sorted(CHUNKS_DIR.glob("*.md")):
        text = chunk_path.read_text(encoding="utf-8").strip()
        stem = chunk_path.stem
        doc_id = stem.rsplit("_", 1)[0]
        split_idx = stem.rsplit("_", 1)[1] if "_" in stem else None

        meta = kb_by_id.get(doc_id, {})
        rows.append(
            {
                "doc_id": doc_id,
                "text": text,
                "plant": meta.get("plant", ""),
                "disease": meta.get("disease", ""),
                "url": meta.get("url", ""),
                "title": meta.get("title")
                or f"{meta.get('plant','')} {meta.get('disease','')}".strip(),
                "section": meta.get("section"),
                "lang": meta.get("lang"),
                "split_idx": meta.get("split_idx", split_idx),
                "n_tokens": meta.get("n_tokens") or len(text.split()),
            }
        )
    return rows


def main() -> None:
    if not KB_PATH.exists():
        raise FileNotFoundError(f"kb.json not found at {KB_PATH}")
    if not CHUNKS_DIR.exists():
        raise FileNotFoundError(f"chunks dir not found at {CHUNKS_DIR}")

    kb_by_id = load_kb()
    rows = build_manifest_rows(kb_by_id)

    df = pd.DataFrame(rows)
    # Normalize types for parquet
    df["split_idx"] = pd.to_numeric(df["split_idx"], errors="coerce").astype("Int64")
    df["title"] = df["title"].fillna("").astype(str)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    url_nonempty = int((df["url"].fillna("") != "").sum())
    print(f"wrote {len(df)} rows to {OUT_PATH} (urls populated: {url_nonempty})")


if __name__ == "__main__":
    main()
