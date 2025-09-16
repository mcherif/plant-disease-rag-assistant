r"""
Unified KB builder for PlantVillage (+ Wikipedia soon).

- Reads PlantVillage KB JSON (from refresh_kb_descriptions.py)
- Normalizes sections to markdown, sanitizes minor noise
- Chunks sentence-aware with overlap
- Optional MinHash/LSH dedup
- Writes chunks/ and a manifest (Parquet or CSV fallback)

Usage:
  python -m src.ingestion.build_kb --sources plantvillage --out data\\kb --min_tokens 50 --max_tokens 1000 --overlap 100 --dedup minhash --dedup-threshold 0.9 --verbose
"""
from src.ingestion.kb_validator import validate_kb
import argparse
import datetime as dt
import json
import re
import time
import requests
import urllib.parse
import urllib.robotparser
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict, Union
from bs4 import BeautifulSoup

WIKI_BASE = "https://{lang}.wikipedia.org"
WIKI_SUMMARY = "/api/rest_v1/page/summary/{title}"
USER_AGENT = "PlantDiseaseKB/0.1 (+https://github.com/mcherif/plant-disease-rag-assistant; https://huggingface.co/spaces/mcherif/Plant-Disease-RAG-Assistant)"

# Canonicalization for common spelling/casing mismatches from PV → Wikipedia
WIKI_NORMALIZE: Dict[str, str] = {
    "haunglongbing (citrus greening)": "huanglongbing",
    "haunglongbing": "huanglongbing",
    "citrus greening": "huanglongbing",
    "citrus greening disease": "huanglongbing",
    "leaf mold": "leaf mold",  # ensure US spelling; UK: 'leaf mould'
    "leaf mould": "leaf mold",
    "leaf mold (tomato)": "tomato leaf mold",
    "tomato leaf mould": "tomato leaf mold",
    "tomato yellow leaf curl virus": "tomato yellow leaf curl virus",
    "northern leaf blight": "northern corn leaf blight",
    "northern leaf blight (corn (maize))": "northern corn leaf blight",
    "cercospora leaf spot gray leaf spot": "gray leaf spot",
    "cercospora leaf spot grey leaf spot": "corn grey leaf spot",
    "leaf blight (isariopsis leaf spot)": "isariopsis leaf spot",
    "spider mites two-spotted spider mite": "two-spotted spider mite",
    "tomato septoria leaf spot": "septoria lycopersici",
    "septoria leaf spot (tomato)": "septoria lycopersici",
    "leaf scorch (strawberry)": "strawberry leaf scorch",
    "target spot": "corynespora cassiicola",
    "tomato target spot": "corynespora cassiicola",
    "esca (black measles)": "esca",
}


class Doc(TypedDict, total=False):
    doc_id: str
    url: str
    title: str
    plant: Optional[str]
    disease: Optional[str]
    section: Optional[str]
    lang: str
    crawl_date: str  # ISO date
    text: str


def make_session() -> requests.Session:
    """Requests session with retry/backoff and custom User-Agent."""
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    sess = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("HEAD", "GET"),
    )
    sess.mount("http://", HTTPAdapter(max_retries=retry))
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    sess.headers.update({"User-Agent": USER_AGENT})
    return sess


class Robots:
    """Cache robots.txt decisions per host."""

    def __init__(self) -> None:
        self._cache: Dict[str, urllib.robotparser.RobotFileParser] = {}

    def allowed(self, url: str, ua: str = USER_AGENT) -> bool:
        parsed = urllib.parse.urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        rp = self._cache.get(host)
        if rp is None:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(urllib.parse.urljoin(host, "/robots.txt"))
            try:
                rp.read()
            except Exception:
                return True  # fail-open to avoid hard blocks in offline runs
            self._cache[host] = rp
        return rp.can_fetch(ua, url)


class RateLimiter:
    """Simple per-host rate limiter (min_interval seconds between calls)."""

    def __init__(self, min_interval: float = 1.0) -> None:
        self.min_interval = min_interval
        self._next: Dict[str, float] = {}

    def wait(self, url: str) -> None:
        now = time.time()
        parsed = urllib.parse.urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        t_next = self._next.get(host, 0.0)
        if now < t_next:
            time.sleep(t_next - now)
        self._next[host] = time.time() + self.min_interval


def fetch_wikipedia_summary(
    title: str,
    lang: str,
    sess: requests.Session,
    robots: Robots,
    rl: RateLimiter,
    timeout: float = 10.0,
) -> Optional[Dict]:
    """Fetch REST summary for a title; returns dict with text, url, title or None on miss."""
    base = WIKI_BASE.format(lang=lang)
    path = WIKI_SUMMARY.format(title=urllib.parse.quote(title, safe=""))
    url = urllib.parse.urljoin(base, path) + "?redirect=true"
    if not robots.allowed(url, ua=USER_AGENT):
        return None
    rl.wait(url)
    resp = sess.get(url, timeout=timeout)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    data = resp.json()
    extract = (data or {}).get("extract") or ""
    if not extract:
        return None
    page_url = (
        (((data or {}).get("content_urls") or {}).get("desktop") or {}).get("page")
        or f"{base}/wiki/{urllib.parse.quote(data.get('title') or title)}"
    )
    norm_title = (data or {}).get("title") or title
    return {"text": extract, "url": page_url, "title": norm_title}


def fetch_wikipedia_intro_action(
    title: str,
    lang: str,
    sess: requests.Session,
    rl: RateLimiter,
    timeout: float = 10.0,
) -> Optional[Dict]:
    """
    Fetch the lead intro via MediaWiki Action API (exintro, plaintext), following redirects.
    Returns dict with text, url, title or None on miss/disambiguation.
    """
    base = WIKI_BASE.format(lang=lang)
    api_url = urllib.parse.urljoin(base, "/w/api.php")
    params = {
        "action": "query",
        "prop": "extracts|info|pageprops",
        "ppprop": "disambiguation",
        "exintro": "1",
        "explaintext": "1",
        "redirects": "1",
        "inprop": "url",
        "format": "json",
        "formatversion": "2",
        "titles": title,
    }
    rl.wait(api_url)
    resp = sess.get(api_url, params=params, timeout=timeout)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    data = resp.json()
    pages = (data.get("query", {}) or {}).get("pages") or []
    if not pages:
        return None
    page = pages[0]
    # Skip missing or disambiguation pages
    if page.get("missing") or ("pageprops" in page and "disambiguation" in (page["pageprops"] or {})):
        return None
    extract = (page.get("extract") or "").strip()
    if not extract:
        return None
    page_url = page.get("canonicalurl") or page.get(
        "fullurl") or f"{base}/wiki/{urllib.parse.quote(page.get('title') or title)}"
    norm_title = page.get("title") or title
    return {"text": extract, "url": page_url, "title": norm_title}


def fetch_wikipedia_search_title(
    query: str,
    lang: str,
    sess: requests.Session,
    rl: RateLimiter,
    timeout: float = 10.0,
) -> Optional[str]:
    """Use MediaWiki search to find the best title for a query; returns normalized title or None."""
    base = WIKI_BASE.format(lang=lang)
    api_url = urllib.parse.urljoin(base, "/w/api.php")
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": "1",
        "srinfo": "totalhits",
        "srprop": "",
        "format": "json",
        "formatversion": "2",
    }
    rl.wait(api_url)
    resp = sess.get(api_url, params=params, timeout=timeout)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    data = resp.json()
    items = (data.get("query", {}) or {}).get("search") or []
    if not items:
        return None
    return items[0].get("title") or None


def _normalize_candidate(cand: str) -> str:
    key = cand.strip().lower()
    return WIKI_NORMALIZE.get(key, cand)


def _wiki_is_relevant(title: str, extract: str, plant: str, disease: str) -> bool:
    """Heuristic filter to avoid lists/off-topic pages from search fallback."""
    t = (title or "").lower()
    ex = (extract or "").lower()
    pl = (plant or "").lower()
    dz = (disease or "").lower()

    # Avoid list/overview pages by default
    if t.startswith("list of"):
        return False

    # If disease is specific, require it in title or extract
    if dz and (dz in t or dz in ex):
        return True

    # For generic disease terms, require plant mention
    generic_terms = ["blight", "mildew", "rust",
                     "leaf spot", "leaf mold", "mite", "virus"]
    if any(g in dz for g in generic_terms):
        return bool(pl) and (pl in t or pl in ex)

    # Fallback: accept if both plant and any generic term appear in text
    return bool(pl) and (pl in ex) and any(g in ex for g in generic_terms)


def _seed_pairs_from_pv(kb_path: Union[str, Path]) -> List[Tuple[str, str]]:
    """Derive (plant, disease) pairs from PlantVillage KB JSON (skip 'healthy')."""
    kb = json.loads(Path(kb_path).read_text(encoding="utf-8"))
    pairs: List[Tuple[str, str]] = []
    seen = set()
    for plant, diseases in kb.items():
        for disease in diseases.keys():
            if disease.lower() == "healthy":
                continue
            key = (plant, disease)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    return pairs


def load_docs_from_wikipedia(
    plants_diseases: List[Tuple[str, str]], lang: str = "en", delay: float = 0.25, verbose: bool = False
) -> List[Doc]:
    """Fetch summaries for candidate titles derived from (plant, disease) pairs using the Action API."""
    sess = make_session()
    rl = RateLimiter(min_interval=delay)
    docs: List[Doc] = []
    total = len(plants_diseases)
    misses = 0
    for plant, disease in plants_diseases:
        # Try multiple title variants; keep order, drop duplicates
        raw_candidates = [
            disease,
            f"{disease} ({plant})",
            f"{plant} {disease}",
        ]
        seen_titles = set()
        candidates = []
        for c in raw_candidates:
            # Normalize common mismatches before querying
            t = _normalize_candidate(c)
            if not t:
                continue
            k = t.casefold()
            if k in seen_titles:
                continue
            seen_titles.add(k)
            candidates.append(t)

        found = False
        for cand in candidates:
            try:
                res = fetch_wikipedia_intro_action(
                    cand, lang=lang, sess=sess, rl=rl)
            except Exception as e:
                if verbose:
                    print(f"[wiki][error] {cand} -> {e}")
                res = None

            # Fallback: search best title, then fetch intro
            if not res:
                try:
                    best = fetch_wikipedia_search_title(
                        cand, lang=lang, sess=sess, rl=rl)
                except Exception as e:
                    if verbose:
                        print(f"[wiki][search-error] {cand} -> {e}")
                    best = None
                if best and best.casefold() != cand.casefold():
                    if verbose:
                        print(f"[wiki][search-hit] {cand} -> {best}")
                    try:
                        res = fetch_wikipedia_intro_action(
                            best, lang=lang, sess=sess, rl=rl)
                    except Exception as e:
                        if verbose:
                            print(f"[wiki][error] {best} -> {e}")
                        res = None

            if not res:
                if verbose:
                    print(f"[wiki][miss-cand] {cand}")
                continue

            # Relevance filter to drop off-topic hits (e.g., lists, cultivars)
            if not _wiki_is_relevant(res["title"], res["text"], plant, disease):
                if verbose:
                    print(f"[wiki][drop] {res['title']} (irrelevant)")
                continue

            text = normalize_to_markdown(res["text"])
            text = _strip_leading_title_repeat(text, disease)
            docs.append(
                {
                    "doc_id": str(uuid.uuid4()),
                    "url": res["url"],
                    "title": res["title"] or f"{plant} — {disease}",
                    "plant": plant,
                    "disease": disease,
                    "section": "wikipedia:summary",
                    "lang": lang,
                    "crawl_date": now_date(),
                    "text": text,
                }
            )
            found = True
            break

        if not found:
            misses += 1
            if verbose:
                print(f"[wiki][miss] {plant} / {disease}")

    if verbose:
        print(
            f"[wiki] tried {total} pairs; loaded {len(docs)} docs; misses {misses}")
    return docs


def load_docs_from_wikipedia_seeded(
    kb_path: Union[str, Path], lang: str = "en", min_interval: float = 0.5, verbose: bool = False
) -> List[Doc]:
    """Seed Wikipedia lookups from PV KB pairs, then fetch summaries."""
    pairs = _seed_pairs_from_pv(kb_path)
    if verbose:
        print(f"[wiki] seeding {len(pairs)} (plant, disease) pairs from PV KB")
    docs = load_docs_from_wikipedia(
        pairs, lang=lang, delay=min_interval, verbose=verbose)
    if verbose:
        print(f"[wiki] loaded {len(docs)} docs from Wikipedia")
    return docs


def parse_args() -> argparse.Namespace:
    """CLI for selecting sources, output directory, chunking, and dedup options."""
    ap = argparse.ArgumentParser(
        description="Build a normalized, chunked KB from PlantVillage (+Wikipedia soon)."
    )
    ap.add_argument(
        "--sources",
        default="plantvillage",
        help="Comma-separated sources: plantvillage,wikipedia",
    )
    ap.add_argument(
        "--out",
        default=str(Path("data") / "kb"),
        help="Output directory for chunks/ and manifest",
    )
    ap.add_argument(
        "--min_tokens", type=int, default=50, help="Minimum tokens per chunk (approx.)"
    )
    ap.add_argument(
        "--max_tokens", type=int, default=1000, help="Maximum tokens per chunk (approx.)"
    )
    ap.add_argument(
        "--overlap", type=int, default=100, help="Token overlap between chunks (approx.)"
    )
    ap.add_argument(
        "--kb",
        default=str(Path("data") / "plantvillage_kb.json"),
        help="Path to PlantVillage KB JSON (produced by refresh_kb_descriptions.py)",
    )
    ap.add_argument(
        "--dedup",
        default="none",
        choices=["none", "minhash"],
        help="Deduplicate chunks (minhash uses LSH; requires 'datasketch')",
    )
    ap.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.9,
        help="MinHash Jaccard threshold for near-duplicate chunks (0.8–0.95 typical)",
    )
    ap.add_argument("--wiki-lang", default="en",
                    help="Wikipedia language code (e.g., en, fr)")
    ap.add_argument(
        "--wiki-interval",
        type=float,
        default=0.5,
        help="Seconds between Wikipedia requests per host (polite rate limit)",
    )
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def now_date() -> str:
    return dt.date.today().isoformat()


def word_tokens(text: str) -> List[str]:
    return re.findall(r"\w+|\S", text, flags=re.UNICODE)


def count_tokens(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def sentence_split(text: str) -> List[str]:
    """Very simple sentence splitter; replace with spaCy/NLTK for better accuracy."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(0-9])", text.strip())
    return [p.strip() for p in parts if p and not p.isspace()]


def normalize_to_markdown(text: str) -> str:
    """Light normalization placeholder; collapse stray whitespace/newlines."""
    return re.sub(r"\s+\n", "\n", text.strip())


def _collapse_repeated_word_labels(text: str) -> str:
    """
    Collapse repeated single-word labels (e.g., 'Bacterium Fungus ...') while preserving order.
    Only applies to lines without punctuation to avoid mangling prose.
    """
    lines = text.splitlines()
    out_lines: List[str] = []
    for line in lines:
        # Skip lines that look like sentences (contain punctuation)
        if re.search(r"[.,;:()\-–—/]", line):
            out_lines.append(line)
            continue
        words = re.findall(r"[A-Za-z]+", line)
        if not words:
            out_lines.append(line)
            continue
        # Apply only if there are duplicates and mostly single-word labels
        if len(set(w.lower() for w in words)) < len(words):
            seen = set()
            uniq = []
            for w in words:
                wl = w.lower()
                if wl in seen:
                    continue
                seen.add(wl)
                uniq.append(w)
            out_lines.append(", ".join(uniq))
        else:
            out_lines.append(line)
    return "\n".join(out_lines).strip()


def _sanitize_cause(text: str) -> str:
    """Sanitize 'Cause' sections (collapse repeated labels)."""
    return _collapse_repeated_word_labels(text)


def _strip_leading_title_repeat(text: str, disease: str) -> str:
    """Remove a leading line that repeats the disease title (e.g., 'Late Blight')."""
    if not text or not disease:
        return text
    lines = text.strip().splitlines()
    if not lines:
        return text
    first = lines[0].strip().strip("—:- ").casefold()
    dz = disease.strip().strip("—:- ").casefold()
    if first.startswith(dz):
        return "\n".join(lines[1:]).lstrip()
    return text


def chunk_text_sentence_aware(
    text: str, max_tokens: int, overlap_tokens: int
) -> List[str]:
    """Pack sentences into chunks up to max_tokens, carrying overlap from the end of the previous chunk."""
    if not text:
        return []
    sents = sentence_split(text)
    chunks: List[str] = []
    cur: List[str] = []
    cur_tok = 0
    for s in sents:
        st = count_tokens(s)
        if cur and cur_tok + st > max_tokens:
            chunks.append(" ".join(cur).strip())
            # Build overlap tail
            if overlap_tokens > 0:
                tail = []
                tok_sum = 0
                for ss in reversed(cur):
                    tok_sum += count_tokens(ss)
                    tail.append(ss)
                    if tok_sum >= overlap_tokens:
                        break
                cur = list(reversed(tail))
                cur_tok = sum(count_tokens(ss) for ss in cur)
            else:
                cur = []
                cur_tok = 0
        cur.append(s)
        cur_tok += st
    if cur:
        chunks.append(" ".join(cur).strip())
    return [c for c in chunks if count_tokens(c) >= 1]


def ensure_out_dirs(out_dir: Union[str, Path]) -> Tuple[Path, Path]:
    """Create the output directory and chunks/ subfolder."""
    out_dir = Path(out_dir)
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, chunks_dir


def write_manifest(rows: List[Dict], out_dir: Path, verbose: bool = False) -> Path:
    """Write a manifest to Parquet if possible; fall back to CSV. Returns the manifest path."""
    out_parquet = out_dir / "manifest.parquet"
    out_csv = out_dir / "manifest.csv"
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows)
        # Ensure column order
        cols = [
            "doc_id",
            "url",
            "title",
            "plant",
            "disease",
            "section",
            "lang",
            "crawl_date",
            "split_idx",
            "text",
            "n_tokens",
        ]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        df = df[cols]
        df.to_parquet(out_parquet, index=False)
        if verbose:
            print(f"[manifest] wrote {out_parquet}")
        return out_parquet
    except Exception as e:
        if verbose:
            print(
                f"[manifest] parquet unavailable ({e}); writing CSV fallback")
        try:
            import csv

            with out_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "doc_id",
                        "url",
                        "title",
                        "plant",
                        "disease",
                        "section",
                        "lang",
                        "crawl_date",
                        "split_idx",
                        "text",
                        "n_tokens",
                    ],
                )
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            if verbose:
                print(f"[manifest] wrote {out_csv}")
            return out_csv
        except Exception as e2:
            raise RuntimeError(f"Failed to write manifest: {e2}") from e2


def write_chunk_file(chunks_dir: Path, doc_id: str, split_idx: int, text: str) -> Path:
    """Write a single chunk to chunks/{doc_id}_{split_idx}.md and return its path."""
    safe_id = re.sub(r"[^A-Za-z0-9_\-]", "_", doc_id)
    path = chunks_dir / f"{safe_id}_{split_idx:04d}.md"
    with path.open("w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")
    return path


def deduplicate_chunks(
    rows: List[Dict],
    method: str,
    threshold: float = 0.9,
    verbose: bool = False,
) -> List[Dict]:
    """Near-duplicate removal using MinHash/LSH. Keeps the first chunk in each near-dup group."""
    if method == "none":
        return rows
    try:
        from datasketch import MinHash, MinHashLSH  # type: ignore
    except Exception as e:
        if verbose:
            print(f"[dedup] datasketch unavailable ({e}); skipping dedup")
        return rows

    def shingles(text: str, n: int = 5) -> List[str]:
        toks = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if len(toks) <= n:
            return [" ".join(toks)] if toks else []
        return [" ".join(toks[i: i + n]) for i in range(0, len(toks) - n + 1)]

    num_perm = 64
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: List[Dict] = []
    mh_index: Dict[str, MinHash] = {}

    def row_key(r: Dict) -> str:
        return f"{r.get('doc_id','')}_{int(r.get('split_idx', 0))}"

    dup_count = 0
    for r in rows:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        mh = MinHash(num_perm=num_perm)
        for g in shingles(text):
            mh.update(g.encode("utf-8"))
        # Query LSH for near-dups
        candidates = lsh.query(mh)
        is_dup = False
        for cid in candidates:
            # Final check using MinHash estimate to be safe
            if mh.jaccard(mh_index[cid]) >= threshold:
                is_dup = True
                break
        if is_dup:
            dup_count += 1
            continue
        kid = row_key(r)
        lsh.insert(kid, mh)
        mh_index[kid] = mh
        kept.append(r)

    if verbose:
        print(
            f"[dedup] kept {len(kept)} / {len(rows)} chunks (removed {dup_count})")
    return kept


def load_docs_from_plantvillage(
    kb_path: Union[str, Path], verbose: bool = False
) -> List[Doc]:  # <-- FIXED
    """Load docs from PlantVillage KB JSON and synthesize markdown sections."""
    kb_path = Path(kb_path)
    if not kb_path.exists():
        raise FileNotFoundError(
            f"PlantVillage KB not found: {kb_path}. Run refresh_kb_descriptions.py first."
        )
    with kb_path.open("r", encoding="utf-8") as f:
        kb = json.load(f)

    docs: List[Doc] = []
    crawl_date = now_date()
    for plant, diseases in kb.items():
        for disease, entry in diseases.items():
            print(f"Processing: plant={plant}, disease={disease}")  # <--- Add this
            if plant == "Apple" and disease.lower() == "apple scab":
                print("DEBUG: Apple scab entry:", entry)
            entry.pop("html", None)
            if disease.lower() == "healthy":
                continue
            desc = (entry or {}).get("description") or ""
            symptoms = (entry or {}).get("symptoms") or ""
            cause = (entry or {}).get("cause") or ""
            url = (entry or {}).get("source") or ""
            title = f"{plant} — {disease}"
            # Sanitize known noisy sections
            if cause:
                cause = _sanitize_cause(cause)
            if desc:
                desc = _strip_leading_title_repeat(desc, disease)
            parts = []
            if desc:
                parts.append(f"# {title}\n\n{desc}")
            if symptoms:
                parts.append(f"## Symptoms\n\n{symptoms}")
            if cause:
                parts.append(f"## Cause\n\n{cause}")
            full_text = normalize_to_markdown("\n\n".join(parts).strip())
            if not full_text:
                continue
            html = entry.get("html")
            management = ""
            if html:
                disease_sections = extract_disease_sections(html)
                # Try to match the current disease name (case-insensitive, partial match)
                for section in disease_sections:
                    if disease.lower() in section.get("disease", "").lower():
                        management = section.get("management", "")
                        break

            doc: Doc = {
                "doc_id": str(uuid.uuid4()),
                "url": url,
                "title": title,
                "plant": plant,
                "disease": disease,
                "section": None,
                "lang": "en",  # TODO: optional lang detection
                "crawl_date": crawl_date,
                "text": full_text,
                "management": management,
            }
            docs.append(doc)
    if verbose:
        print(f"[pv] loaded {len(docs)} docs from {kb_path}")
    return docs

# --- Add/Update: Structured field extraction from raw text chunks ---


def extract_structured_fields(text: str) -> dict:
    """
    Extracts symptoms, cause, management, and prevention sections from a markdown or plain text chunk.
    Returns a dict with keys: symptoms, cause, management, prevention.
    """
    # Simple regex-based extraction for markdown headings (e.g., ## Symptoms)
    sections = {
        "symptoms": "",
        "cause": "",
        "management": "",
        "prevention": ""
    }
    current = None
    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^(#+\s*)?symptoms[:]?$", line, re.IGNORECASE):
            current = "symptoms"
            continue
        elif re.match(r"^(#+\s*)?cause[:]?$", line, re.IGNORECASE):
            current = "cause"
            continue
        elif re.match(r"^(#+\s*)?(management|treatment|control)[:]?$", line, re.IGNORECASE):
            current = "management"
            continue
        elif re.match(r"^(#+\s*)?prevention[:]?$", line, re.IGNORECASE):
            current = "prevention"
            continue
        elif re.match(r"^#+\s*\w+", line):  # Any other heading resets
            current = None
            continue
        if current:
            sections[current] += (line + " ")
    # Clean up whitespace
    for k in sections:
        sections[k] = sections[k].strip()
    return sections


def extract_structured_fields_html(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    sections = {
        "symptoms": "",
        "cause": "",
        "management": "",
        "prevention": ""
    }
    # Example: find divs by class
    for key in sections:
        div = soup.find("div", class_=key)
        if div:
            sections[key] = div.get_text(separator=" ", strip=True)
    return sections

# Example usage in your KB build loop:
# for chunk in chunks:
#     fields = extract_structured_fields(chunk["text"])
#     chunk["symptoms"] = fields["symptoms"]
#     chunk["cause"] = fields["cause"]
#     chunk["management"] = fields["management"]
#     chunk["prevention"] = fields["prevention"]
#     # ...existing code to save chunk...

# This will populate the structured fields for every KB entry if the text contains recognizable sections.


def save_kb_and_validate(kb, out_path):
    import json
    # Save KB
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)
    # Validate KB
    report = validate_kb(kb)
    if report:
        print(
            f"KB validation found {len(report)} incomplete/problematic entries. See validation_report.json for details.")
        with open("validation_report.json", "w", encoding="utf-8") as vf:
            json.dump(report, vf, ensure_ascii=False, indent=2)
    else:
        print("KB validation passed: all entries complete.")

# Replace your final KB save step with:
# save_kb_and_validate(kb, out_path)

# This will run validation after building the KB and output a report for review or targeted scraping.


def build_kb(
    sources: List[str],
    out_dir: Union[str, Path],
    kb_path: Union[str, Path],
    min_tokens: int,
    max_tokens: int,
    overlap: int,
    dedup_method: str,
    dedup_threshold: float = 0.9,
    wiki_lang: str = "en",
    wiki_interval: float = 0.5,
    verbose: bool = False,
) -> Path:
    """Main pipeline: load docs, chunk, optionally dedup, and write manifest."""
    out_dir, chunks_dir = ensure_out_dirs(out_dir)

    # Load documents
    docs: List[Doc] = []
    srcs = [s.strip().lower() for s in sources if s.strip()]
    if "plantvillage" in srcs:
        docs.extend(load_docs_from_plantvillage(kb_path, verbose=verbose))
    if "wikipedia" in srcs:
        docs.extend(
            load_docs_from_wikipedia_seeded(
                kb_path, lang=wiki_lang, min_interval=wiki_interval, verbose=verbose
            )
        )

    if not docs:
        raise RuntimeError("No documents loaded. Check --sources and inputs.")

    # Chunk
    rows: List[Dict] = []
    for d in docs:
        text = d["text"]
        chunks = chunk_text_sentence_aware(
            text, max_tokens=max_tokens, overlap_tokens=overlap
        )
        # Ensure min_tokens
        chunks = [c for c in chunks if count_tokens(c) >= min_tokens]
        if d["plant"] == "Apple" and d["disease"].lower() == "apple scab":
            print("DEBUG: Before chunking:", d)
        for i, ch in enumerate(chunks):
            n_tokens = count_tokens(ch)
            row = {
                "doc_id": d["doc_id"],
                "url": d["url"],
                "title": d["title"],
                "plant": d.get("plant"),
                "disease": d.get("disease"),
                "section": d.get("section"),
                "lang": d.get("lang", "en"),
                "crawl_date": d.get("crawl_date", now_date()),
                "split_idx": i,
                "text": ch,
                "n_tokens": n_tokens,
            }
            rows.append(row)
            write_chunk_file(chunks_dir, d["doc_id"], i, ch)
            if row['plant'] == "Apple" and row['disease'].lower() == "apple scab":
                print("DEBUG: Apple scab chunk:", row)
    if verbose:
        print(f"[chunk] wrote {len(rows)} chunks to {chunks_dir}")

    # Deduplicate (stub)
    rows = deduplicate_chunks(
        rows, method=dedup_method, threshold=dedup_threshold, verbose=verbose
    )

    # Prepare KB for validation (add structured fields to each row)
    for row in rows:
        fields = extract_structured_fields(row["text"])
        row["symptoms"] = fields["symptoms"]
        row["cause"] = fields["cause"]
        row["management"] = fields["management"]
        row["prevention"] = fields["prevention"]
        # Optionally, add references if available

    # After building your KB rows (before saving):
    for row in rows:
        if "html" in row:
            del row["html"]

    # Manifest
    manifest_path = write_manifest(rows, out_dir, verbose=verbose)

    # Save and validate KB (after manifest_path is assigned)
    save_kb_and_validate(rows, str(out_dir / "kb.json"))

    return manifest_path


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    sources = args.sources.split(",") if args.sources else []
    try:
        manifest = build_kb(
            sources=sources,
            out_dir=args.out,
            kb_path=args.kb,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            overlap=args.overlap,
            dedup_method=args.dedup,
            dedup_threshold=args.dedup_threshold,
            wiki_lang=args.wiki_lang,
            wiki_interval=args.wiki_interval,
            verbose=args.verbose,
        )
        if args.verbose:
            print(f"[done] manifest at {manifest}")
        return 0
    except Exception as e:
        print(f"[error] {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


def extract_disease_sections(html: str) -> list:
    """
    Extracts all disease sections and their management from PlantVillage HTML.
    Returns a list of dicts: {disease, symptoms, cause, management, prevention}
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []
    current = {}
    for tag in soup.find_all(['h4', 'div']):
        if tag.name == 'h4':
            # Save previous disease section if exists
            if current:
                results.append(current)
            # Start new disease section
            disease_name = tag.get_text(separator=" ", strip=True)
            current = {"disease": disease_name}
        elif tag.name == 'div' and tag.get('class'):
            cls = tag.get('class')[0]
            current[cls] = tag.get_text(separator=" ", strip=True)
    # Save last section
    if current:
        results.append(current)
    return results

# Example usage:
# disease_sections = extract_disease_sections(html)
# for section in disease_sections:
#     if "Powdery mildew" in section["disease"]:
#         print("Management:", section.get("management", ""))


def extract_disease_info(html):
    soup = BeautifulSoup(html, "html.parser")
    diseases = []
    # Find all disease blocks
    for disease_div in soup.find_all("div", id=lambda x: x and x.startswith("disease-")):
        disease = {}
        # Title
        h4 = disease_div.find_next("h4")
        if h4:
            disease["title"] = h4.get_text(strip=True)
            # Scientific name (italic)
            sci = h4.find("i")
            if sci:
                disease["scientific_name"] = sci.get_text(strip=True)
        # Symptoms
        symptoms = disease_div.find_next("div", class_="symptoms")
        if symptoms:
            disease["symptoms"] = symptoms.get_text(strip=True)
        # Cause
        cause = disease_div.find_next("div", class_="cause")
        if cause:
            disease["cause"] = cause.get_text(strip=True)
        # Comments
        comments = disease_div.find_next("div", class_="comments")
        if comments:
            disease["comments"] = comments.get_text(strip=True)
        # Management
        management = disease_div.find_next("div", class_="management")
        if management:
            disease["management"] = management.get_text(strip=True)
        # Images (optional)
        images = []
        for img in disease_div.find_all("img"):
            images.append({
                "url": img.get("src"),
                "caption": img.find_next("div", class_="image_caption").get_text(strip=True) if img.find_next("div", class_="image_caption") else ""
            })
        if images:
            disease["images"] = images
        diseases.append(disease)
    return diseases


# Example usage
with open("data/disease_and_pests.html", encoding="utf-8") as f:
    html = f.read()
diseases = extract_disease_info(html)
with open("data/apple_diseases.json", "w", encoding="utf-8") as f:
    json.dump(diseases, f, indent=2, ensure_ascii=False)


# Example: extract text after <h4>Management</h4>
def extract_management_text(disease_div):
    h4 = disease_div.find(lambda tag: tag.name == "h4" and "management" in tag.text.lower())
    if h4:
        # Get the next sibling paragraph
        p = h4.find_next_sibling("p")
        if p:
            return p.get_text(strip=True)
    return ""
