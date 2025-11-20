"""PlantVillage KB refresher.

Purpose:
- Update per-disease fields (summary/description, symptoms, cause) in data/plantvillage_kb.json
  by scraping PlantVillage topic 'infos' pages and disease pages.

How it works:
- For each (plant, disease) in the KB, fetch the PlantVillage topic page(s).
- Prefer scoped extraction from the disease section under the plant's 'infos' page.
- Fall back to disease page URLs and page-level heuristics; optional Google CSE lookup.
- Write the refreshed KB to --out (or overwrite --in when omitted).

Examples:
  # Preview to a new file (only fill blanks)
  python -m src.ingestion.refresh_kb_descriptions --in data\\plantvillage_kb.json --out data\\plantvillage_kb.updated.json --only-empty --max-sentences 2 --verbose

  # Refresh in place (overwrite)
  python -m src.ingestion.refresh_kb_descriptions --in data\\plantvillage_kb.json --out data\\plantvillage_kb.json --force --max-sentences 2 --allow-google --verbose

Notes:
- Set GOOGLE_API_KEY and GOOGLE_CSE_ID to enable the Google CSE fallback.
"""
import argparse
import json
import os
import re
import time
from typing import Optional, Tuple, Dict, List, Any
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from difflib import SequenceMatcher

USER_AGENT = "plant-disease-rag-assistant/0.3 (+https://github.com/mcherif/plant-disease-rag-assistant)"

# Known tricky PlantVillage topic aliases
PV_TOPIC_ALIASES = {
    "Corn (maize)": ["corn-maize", "maize", "corn"],
    "Cherry (including sour)": ["cherry-including-sour", "cherry"],
    "Pepper, bell": ["pepper-bell", "bell-pepper", "pepper"],
    "Olive": ["olive"],
}


def _slugify_topic(name: str) -> str:
    s = (name or "").lower().replace("&", "and")
    s = re.sub(r"[^\w]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def _slugify_disease(name: str) -> str:
    s = (name or "").lower().replace("&", "and")
    s = re.sub(r"\([^)]*\)", " ", s)  # drop parentheticals
    s = re.sub(r"[^\w]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def _topic_slugs_for(plant: str) -> list[str]:
    return PV_TOPIC_ALIASES.get(plant, [_slugify_topic(plant)])


def _guess_pv_disease_urls(topic_slug: str, disease_name: str) -> list[str]:
    ds = _slugify_disease(disease_name)
    # try most common PV patterns
    candidates = [
        f"https://plantvillage.psu.edu/topics/{topic_slug}/diseases-and-pests/{ds}",
        f"https://plantvillage.psu.edu/topics/{topic_slug}/infos/{ds}",
        f"https://plantvillage.psu.edu/topics/{topic_slug}/diseases/{ds}",
    ]
    # add a few relaxed variants (spaces vs hyphens already handled by slugify)
    if "-" in ds:
        short = ds.split("-")[0]
        candidates.extend([
            f"https://plantvillage.psu.edu/topics/{topic_slug}/diseases-and-pests/{short}",
        ])
    dedup = []
    seen = set()
    for u in candidates:
        if u not in seen:
            seen.add(u)
            dedup.append(u)
    return dedup


def _session(timeout=12.0) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.8"})
    retry = Retry(
        total=3,
        backoff_factor=0.4,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"],
        raise_on_status=False,
    )
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))

    orig_request = s.request

    def with_timeout(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return orig_request(method, url, **kwargs)
    s.request = with_timeout  # type: ignore
    return s


def _norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_match(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def _sentences(text: str) -> list[str]:
    # naive sentence split; good enough for short blurbs
    parts = re.split(r"(?<=[\.\!\?])\s+", _norm(text))
    return [p.strip() for p in parts if p.strip()]


def _fetch_html(sess: requests.Session, url: str) -> Optional[BeautifulSoup]:
    try:
        r = sess.get(url, allow_redirects=True)
        if r.status_code != 200:
            return None
        return BeautifulSoup(r.text, "html.parser")
    except requests.RequestException:
        return None


def _extract_summary_from_pv_disease_page(soup: BeautifulSoup, max_sentences: int = 2) -> Optional[str]:
    # Heuristics: look for first meaningful paragraph within main/article/content
    scopes = []
    for sel in ("main", "article", "[role='main']", ".content", ".pv-content", ".post-content", "body"):
        found = soup.select_one(sel)
        if found:
            scopes.append(found)
    scopes = scopes or [soup]

    best = None
    for scope in scopes:
        # Prefer longer first paragraph
        for p in scope.find_all("p"):
            text = _norm(p.get_text(" "))
            if len(text.split()) >= 25:  # threshold for meaningful paragraph
                best = text
                break
        if best:
            break

    if not best:
        # fallback: longest paragraph on page
        candidates = [_norm(p.get_text(" ")) for p in soup.find_all("p")]
        candidates = [c for c in candidates if len(c.split()) >= 20]
        if candidates:
            best = max(candidates, key=lambda t: len(t))

    if not best:
        return None

    sents = _sentences(best)
    if not sents:
        return None
    return " ".join(sents[:max_sentences])


def _extract_section_by_heading(soup: BeautifulSoup, titles: list[str], max_sentences: int = 0, max_chars: int = 1200) -> Optional[str]:
    if not soup:
        return None
    titles_norm = [t.lower() for t in titles]
    # find first heading that matches
    hdr = None
    for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):  # include h5/h6
        label = (h.get_text(" ") or "").strip()
        if not label:
            continue
        if any(t in label.lower() for t in titles_norm):
            hdr = h
            break
    if not hdr:
        return None

    chunks = []
    # walk following siblings until next heading
    sib = hdr
    while True:
        sib = sib.find_next_sibling()
        if sib is None:
            break
        if getattr(sib, "name", None) in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            break
        # include paragraphs and list items
        if sib.name in ["p", "li"]:
            text = _norm(sib.get_text(" "))
            if text:
                chunks.append(text)
        # capture bullet lists compactly
        if sib.name in ["ul", "ol"]:
            for li in sib.find_all("li"):
                t = _norm(li.get_text(" "))
                if t:
                    chunks.append(t)
        # capture explicit blocks like <div class="symptoms">...</div>
        if sib.name == "div":
            classes = " ".join((sib.get("class") or [])).lower()
            if any(key in classes for key in ["symptom", "symptoms"]):
                text = _norm(sib.get_text(" "))
                if text:
                    chunks.append(text)

    if not chunks:
        return None
    text = " ".join(chunks)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]
    if max_sentences and max_sentences > 0:
        sents = _sentences(text)
        if sents:
            text = " ".join(sents[:max_sentences])
    return text or None


def _extract_symptoms_from_pv_disease_page(soup: BeautifulSoup, max_sentences: int = 0) -> Optional[str]:
    # try headings first
    heading_sets = [
        ["Symptoms"],
        ["Signs"],
        ["Signs and symptoms"],
    ]
    for titles in heading_sets:
        out = _extract_section_by_heading(
            soup, titles, max_sentences=max_sentences)
        if out:
            return out
    # direct blocks like <div class="symptoms">...</div>
    blocks = soup.select("div.symptoms, div[class*='symptom']")
    if blocks:
        text = " ".join(_norm(b.get_text(" "))
                        for b in blocks if _norm(b.get_text(" ")))
        return text or None
    return None


def _extract_cause_from_pv_disease_page(soup: BeautifulSoup, max_sentences: int = 0) -> Optional[str]:
    # try headings first
    heading_sets = [
        ["Cause"],
        ["Causes"],
        ["Causal agent", "Causal agents", "Causal organism"],
        ["Pathogen"],
        ["Etiology"],
    ]
    for titles in heading_sets:
        out = _extract_section_by_heading(
            soup, titles, max_sentences=max_sentences)
        if out:
            return out
    # direct blocks like <div class="cause">...</div> or "causal"
    blocks = soup.select("div.cause, div[class*='caus']")
    if blocks:
        text = " ".join(_norm(b.get_text(" "))
                        for b in blocks if _norm(b.get_text(" ")))
        return text or None
    return None

# Extract a compact description, symptoms, cause, and management scoped to the disease h4 section.

# Strategy:
# - Find the matching <h4> for the disease (exact, then fuzzy).
# - Within the section (until next h4), pick a paragraph-like summary (avoid galleries),
#   and collect Symptoms/Cause/Management from classed blocks or following headings.
# - Fallback summary: "Common — Latin" (when available).

# Returns:
#   (summary, symptoms, cause, management) — each Optional[str]


def _extract_inline_for_disease(soup: BeautifulSoup, disease_name: str, max_desc_sentences: int = 2) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:

    # 1) Find the disease header (h4 that starts with the disease name)
    target = None
    header_text = None
    latin_text = None

    def norm(s: str) -> str:
        s = " ".join(s.lower().strip().split())
        s = re.sub(r"[\(\)\[\]\.,;:–—\-_/]+", " ", s)
        return " " .join(s.split())

    # Common corrections/synonyms
    synonyms = {
        "haunglongbing (citrus greening)": "huanglongbing (citrus greening)",
        "cercospora leaf spot gray leaf spot": "gray leaf spot",
        "spider mites two spotted spider mite": "spider mites",
    }

    dn_raw = disease_name.strip()
    dn = norm(dn_raw)
    dn = synonyms.get(dn, dn)

    # Try exact startswith match first
    for h in soup.find_all("h4"):
        text = " ".join(h.get_text(" ", strip=True).split())
        if text.lower().startswith(dn_raw.lower().strip()):
            target = h
            header_text = text
            i_tag = h.find("i")
            if i_tag:
                latin_text = " ".join(i_tag.get_text(" ", strip=True).split())
            break

    # Fallback: fuzzy/select best matching h4
    if not target:
        best = (0.0, None, None, None)  # score, tag, header_text, latin
        for h in soup.find_all("h4"):
            text = " ".join(h.get_text(" ", strip=True).split())
            nt = norm(text)
            # token overlap + similarity
            s1 = set(dn.split())
            s2 = set(nt.split())
            overlap = len(s1 & s2) / max(1, len(s1))
            sim = SequenceMatcher(None, dn, nt).ratio()
            score = max(overlap, sim)
            if score > best[0]:
                i_tag = h.find("i")
                lat = " ".join(i_tag.get_text(
                    " ", strip=True).split()) if i_tag else None
                best = (score, h, text, lat)
        if best[0] >= 0.55:
            target, header_text, latin_text = best[1], best[2], best[3]

    if not target:
        return None, None, None, None

    # 2) Iterate siblings until the next disease h4 (scope)
    def is_disease_heading(tag: Tag) -> bool:
        return isinstance(tag, Tag) and tag.name == "h4"

    # 3) Helpers
    def clean_text(t: str) -> str:
        return " ".join(t.split())

    def take_first_sentences(text: str, n: int) -> str:
        if not n:
            return text
        parts = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(parts[:n]).strip()

    def is_galleryish(tag: Tag) -> bool:
        if not isinstance(tag, Tag):
            return False
        if tag.name in ("script", "style"):
            return True
        tid = (tag.get("id") or "").lower()
        tcl = " ".join((tag.get("class") or [])).lower()
        if tid.startswith("links-disease-"):
            return True
        if "blueimp-gallery" in tcl or "medium-images" in tcl or "image_caption" in tcl:
            return True
        if tag.select_one("a[data-gallery], img"):
            return True
        return False

    summary: Optional[str] = None
    symptoms: Optional[str] = None
    cause: Optional[str] = None
    management: Optional[str] = None

    # 4) Walk the scope
    sib = target.next_sibling
    while sib and not (isinstance(sib, Tag) and is_disease_heading(sib)):
        if isinstance(sib, Tag):
            # Summary: prefer the first real paragraph; avoid gallery-like containers
            if summary is None:
                if sib.name == "p":
                    txt = clean_text(sib.get_text(" ", strip=True))
                    if len(txt) >= 40:
                        summary = txt
                elif sib.name == "div" and not is_galleryish(sib):
                    # only consider divs that look like textual intros
                    classes = " ".join((sib.get("class") or [])).lower()
                    if any(k in classes for k in ("summary", "overview", "intro", "lead", "description")):
                        txt = clean_text(sib.get_text(" ", strip=True))
                        if len(txt) >= 40:
                            summary = txt

            # Direct class-based extraction (most robust for this page)
            classes = sib.get("class", []) if isinstance(sib, Tag) else []
            if symptoms is None and "symptoms" in classes:
                symptoms = clean_text(sib.get_text(" ", strip=True))
            if cause is None and "cause" in classes:
                cause = clean_text(sib.get_text(" ", strip=True))
            if management is None and "management" in classes:
                management = clean_text(sib.get_text(" ", strip=True))

            # Heading-based extraction (h5 like "Symptoms"/"Cause"/"Management")
            if sib.name in ("h5", "h6", "strong", "b"):
                heading_text = clean_text(
                    sib.get_text(" ", strip=True)).lower()
                # Collect content until the next heading of same/greater level or the next disease h4
                if symptoms is None and "symptom" in heading_text:
                    parts = []
                    ptr = sib.next_sibling
                    while ptr and not (isinstance(ptr, Tag) and (ptr.name in ("h5", "h6", "strong", "b") or is_disease_heading(ptr))):
                        if isinstance(ptr, Tag) and ptr.name in ("div", "p", "ul", "ol", "li"):
                            parts.append(clean_text(
                                ptr.get_text(" ", strip=True)))
                        ptr = ptr.next_sibling
                    if parts:
                        symptoms = "\n".join([p for p in parts if p])
                if cause is None and ("cause" in heading_text or "causal agent" in heading_text or "etiology" in heading_text or "pathogen" in heading_text):
                    parts = []
                    ptr = sib.next_sibling
                    while ptr and not (isinstance(ptr, Tag) and (ptr.name in ("h5", "h6", "strong", "b") or is_disease_heading(ptr))):
                        if isinstance(ptr, Tag) and ptr.name in ("div", "p", "ul", "ol", "li"):
                            parts.append(clean_text(
                                ptr.get_text(" ", strip=True)))
                        ptr = ptr.next_sibling
                    if parts:
                        cause = "\n".join([p for p in parts if p])
                if management is None and "management" in heading_text:
                    parts = []
                    ptr = sib.next_sibling
                    while ptr and not (isinstance(ptr, Tag) and (ptr.name in ("h5", "h6", "strong", "b") or is_disease_heading(ptr))):
                        if isinstance(ptr, Tag) and ptr.name in ("div", "p", "ul", "ol", "li"):
                            parts.append(clean_text(
                                ptr.get_text(" ", strip=True)))
                        ptr = ptr.next_sibling
                    if parts:
                        management = "\n".join([p for p in parts if p])
        sib = sib.next_sibling

    # 5) Fallbacks
    if not summary:
        # Prefer a compact "Common — Latin" form if Latin exists
        if latin_text:
            summary = f"{disease_name} — {latin_text}"
        else:
            summary = header_text or disease_name
    if summary and max_desc_sentences:
        summary = take_first_sentences(summary, max_desc_sentences)

    # Heuristic fallback for cause if not captured from class/heading
    if not cause:
        scope_text = []
        sib = target.next_sibling
        while sib and not (isinstance(sib, Tag) and is_disease_heading(sib)):
            if isinstance(sib, Tag) and sib.name in ("p", "div", "li") and not is_galleryish(sib):
                scope_text.append(clean_text(sib.get_text(" ", strip=True)))
            sib = sib.next_sibling
        joined = " ".join(scope_text)
        m = re.search(
            r"(?:caused by|causal agent[:\-]?)\s+([^.;]+)", joined, flags=re.I)
        if m:
            cause = clean_text(m.group(1))

    return summary or None, symptoms or None, cause or None, management or None


# Thin wrapper used by refresh_descriptions (must be defined above it)
def _extract_scoped_sections_from_pv_page(
    soup: BeautifulSoup,
    disease_name: str,
    max_sentences: int = 2,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Wrapper to keep refresh_descriptions decoupled from the extractor details.
    """
    return _extract_inline_for_disease(soup, disease_name, max_desc_sentences=max_sentences)


def _find_pv_disease_link_from_topic(soup: BeautifulSoup, topic_slug: str, disease_name: str) -> Optional[str]:
    # Find links under /topics/<slug>/diseases-and-pests/*
    links = soup.select(f'a[href*="/topics/{topic_slug}/diseases-and-pests/"]')
    if not links:
        # generic fallback: any diseases-and-pests links on page
        links = soup.select('a[href*="/diseases-and-pests/"]')
    if not links:
        return None

    target = _norm_match(disease_name)
    best_url = None
    best_score = -1

    for a in links:
        label = a.get_text(" ") or ""
        label_norm = _norm_match(label)
        # Simple score: overlap of tokens
        set_d = set(target.split())
        set_l = set(label_norm.split())
        score = len(set_d & set_l)
        href = a.get("href")
        if href and score >= best_score:
            best_score = score
            best_url = href

    if not best_url:
        return None
    # Normalize to absolute URL
    if best_url.startswith("/"):
        return f"https://plantvillage.psu.edu{best_url}"
    return best_url


def _find_pv_disease_link_from_infos(soup: BeautifulSoup, topic_slug: str, disease_name: str) -> Optional[str]:
    """
    Inside the infos page (preferably within 'Common Pests and Diseases'), find a link to the disease page.
    """
    if not soup:
        return None
    target = set(_norm_match(disease_name).split())

    section = None
    for h in soup.find_all(["h2", "h3", "h4"]):
        if "common pests and diseases" in (h.get_text(" ") or "").lower():
            section = h.parent if h.parent else h
            break
    scope = section or soup

    for a in scope.find_all("a"):
        href = (a.get("href") or "").strip()
        if not href or "/diseases-and-pests/" not in href:
            continue
        label = _norm_match(a.get_text(" "))
        if not label:
            continue
        if len(set(label.split()) & target) >= min(2, len(target)):
            if href.startswith("/"):
                href = urljoin("https://plantvillage.psu.edu/",
                               href.lstrip("/"))
            return href
    return None


def _google_cse_search(plant: str, disease: str, per_page: int = 3) -> list[str]:
    key = os.environ.get("GOOGLE_API_KEY")
    cx = os.environ.get("GOOGLE_CSE_ID")
    if not key or not cx:
        return []
    try:
        q = f'site:plantvillage.psu.edu "{plant}" "{disease}" diseases and pests'
        r = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": key, "cx": cx, "q": q, "num": per_page},
            timeout=15,
        )
        if r.status_code != 200:
            return []
        data = r.json() or {}
        items = data.get("items") or []
        return [it.get("link") for it in items if it.get("link")]
    except Exception:
        return []


def _collect_pv_disease_links(sess: requests.Session, topic_slug: str) -> list[str]:
    """Collect disease page links from both /infos and /diseases-and-pests listings."""
    links = set()
    bases = [
        f"https://plantvillage.psu.edu/topics/{topic_slug}/infos",
        f"https://plantvillage.psu.edu/topics/{topic_slug}/diseases-and-pests",
        f"https://plantvillage.psu.edu/topics/{topic_slug}/diseases-and-pests/",
    ]
    for base in bases:
        soup = _fetch_html(sess, base)
        if not soup:
            continue
        # anchors pointing to diseases-and-pests pages
        for a in soup.select('a[href*="/diseases-and-pests/"]'):
            href = (a.get("href") or "").strip()
            if not href:
                continue
            if href.startswith("/"):
                href = urljoin("https://plantvillage.psu.edu/",
                               href.lstrip("/"))
            if "/diseases-and-pests/" in href:
                links.add(href)
    return list(links)


def _clean_management(text: str) -> str:
    """
    Cleans the 'management' field by removing redundant 'Management' at the start of the text.
    """
    # Debug: Print the original text
    print(f"[DEBUG] Original management text: {text!r}")

    if text.strip().lower().startswith("management"):
        # Remove the first occurrence of "Management" (case-insensitive) and any leading spaces
        text = text[len("Management"):].strip()
        # Debug: Print after first removal
        print(f"[DEBUG] After removing first 'Management': {text!r}")

        # Ensure no duplicate "Management" remains at the start
        if text.lower().startswith("management"):
            text = text[len("Management"):].strip()
            # Debug: Print after second removal
            print(f"[DEBUG] After removing duplicate 'Management': {text!r}")

    # Debug: Print the final cleaned text
    print(f"[DEBUG] Cleaned management text: {text!r}")
    return text


def _strip_management(text: str) -> str:
    # Remove a leading "management" even if repeated.
    if text.lower().startswith("management"):
        # Remove "management" then any following punctuation/spaces.
        return text[len("management"):].lstrip(" -:;")
    return text


def clean_management_fields(kb: dict) -> None:
    for plant, diseases in kb.items():
        if isinstance(diseases, dict):
            for disease, record in diseases.items():
                if isinstance(record, dict) and "management" in record:
                    record["management"] = _strip_management(
                        record["management"])




def _coerce_kb_structure(data: Any) -> Tuple[dict, Optional[Dict[str, Any]]]:
    """Coerce KB JSON data into the nested {plant: {disease: payload}} format.

    Returns a tuple of (kb_dict, list_meta). list_meta is None when data is already a dict."""
    if isinstance(data, dict):
        return data, None
    if not isinstance(data, list):
        raise ValueError("KB JSON must be a dict or list of records.")
    index_map: Dict[Tuple[str, str], List[int]] = {}
    nested: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        plant = entry.get("plant")
        disease = entry.get("disease")
        if not plant or not disease:
            continue
        index_map.setdefault((plant, disease), []).append(idx)
        payload = nested.setdefault(plant, {}).setdefault(
            disease,
            {
                "source": "",
                "description": "",
                "symptoms": "",
                "cause": "",
                "management": "",
            },
        )

        def _merge(field: str, value: Any) -> None:
            if value and not payload.get(field):
                payload[field] = value

        _merge("source", entry.get("source") or entry.get("url"))
        _merge("description", entry.get("description"))
        _merge("symptoms", entry.get("symptoms"))
        _merge("cause", entry.get("cause"))
        _merge("management", entry.get("management"))
    return nested, {"rows": data, "index_map": index_map}



def _apply_updates_to_rows(list_meta: Dict[str, Any], updated: Dict[str, Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Propagate refreshed fields back to the list-style KB rows."""
    rows = json.loads(json.dumps(list_meta["rows"]))  # deep copy
    index_map: Dict[Tuple[str, str], List[int]] = list_meta["index_map"]
    for (plant, disease), indices in index_map.items():
        plant_block = updated.get(plant)
        if not isinstance(plant_block, dict):
            continue
        payload = plant_block.get(disease)
        if not isinstance(payload, dict):
            continue
        for idx in indices:
            row = rows[idx]
            for field in ("description", "symptoms", "cause", "management"):
                if field in payload:
                    row[field] = payload.get(field, row.get(field, ""))
            src = payload.get("source")
            if src:
                row["source"] = src
                row["url"] = src
    return rows

def refresh_descriptions(
    kb_path: str,
    out_path: Optional[str],
    delay: float,
    max_sentences: int,
    only_empty: bool,
    allow_google: bool,
    verbose: bool = False,
    force: bool = False,
    plant_filter: Optional[str] = None,
    disease_filter: Optional[str] = None,
    clean_management: bool = False,  # new parameter, disabled by default
) -> dict:
    """
    Load the KB JSON, refresh entries by scraping PlantVillage, and write the result.

    Args:
      kb_path: input KB JSON
      out_path: output path (defaults to kb_path if None)
      delay: optional sleep between requests (seconds)
      max_sentences: cap for summary sentences (0 = unlimited)
      only_empty: update only entries missing description (and optionally symptoms/cause)
      allow_google: enable Google CSE fallback for tricky lookups
      verbose: print progress
      force: overwrite existing populated fields
      plant_filter / disease_filter: optional regex filters to limit scope
      clean_management: clean management fields if True

    Returns:
      stats dict with counts: checked, updated, skipped, failed, google_used
    """
    with open(kb_path, "r", encoding="utf-8") as f:
        kb_data = json.load(f)

    kb, list_meta = _coerce_kb_structure(kb_data)
    updated = json.loads(json.dumps(kb))  # deep copy
    sess = _session()
    plant_re = re.compile(plant_filter, re.I) if plant_filter else None
    disease_re = re.compile(disease_filter, re.I) if disease_filter else None

    stats = {"checked": 0, "updated": 0,
             "skipped": 0, "failed": 0, "google_used": 0}

    for plant, diseases in kb.items():
        if not isinstance(diseases, dict):
            continue
        for disease, payload in diseases.items():
            if not isinstance(payload, dict):
                continue
            if plant_re and not plant_re.search(plant):
                continue
            if disease_re and not disease_re.search(disease):
                continue

            src = payload.get("source") or ""
            desc = payload.get("description") or ""

            # Skip classes that have no disease page
            if disease.strip().lower() == "healthy":
                stats["skipped"] += 1
                if verbose:
                    print(f"[SKIP] {plant} / healthy: no disease page")
                continue

            if (only_empty and not force) and desc and not desc.strip().startswith("⚠️"):
                stats["skipped"] += 1
                if verbose:
                    print(
                        f"[SKIP] {plant} / {disease}: already has description")
                continue

            stats["checked"] += 1

            # Try to derive topic slug from source; if missing, try aliases/slugify
            topic_slug = None
            try:
                path = urlparse(src).path
                m = re.search(r"/topics/([^/]+)/", path or "")
                if m:
                    topic_slug = m.group(1)
            except Exception:
                pass
            cand_slugs = [
                topic_slug] if topic_slug else _topic_slugs_for(plant)

            disease_url = None
            inline_applied = False

            # 1) If source already is a disease page, use it
            if "/diseases-and-pests/" in src:
                disease_url = src

            # Prefer the plant's infos page directly (scoped parsing will target the right h4)
            if src and "plantvillage.psu.edu/topics/" in src and src.endswith("/infos"):
                disease_url = src

            # 2) Otherwise search disease links on topic pages (infos + diseases-and-pests)
            if not disease_url:
                for slug in cand_slugs:
                    # a) Try finding a disease link specifically inside the infos page
                    infos_url = f"https://plantvillage.psu.edu/topics/{slug}/infos"
                    soup_infos = _fetch_html(sess, infos_url)
                    link = _find_pv_disease_link_from_infos(
                        soup_infos, slug, disease) if soup_infos else None
                    if not link:
                        # b) Fallback to generic anchor matching on the infos page
                        link = _find_pv_disease_link_from_topic(
                            soup_infos, slug, disease) if soup_infos else None
                    if link:
                        disease_url = link
                        break

                # c) guess canonical disease URLs from the disease name and test them
                if not disease_url:
                    for slug in cand_slugs:
                        for guess in _guess_pv_disease_urls(slug, disease):
                            try:
                                r = sess.get(guess, allow_redirects=True)
                                if r.status_code == 200 and "/diseases" in r.url:
                                    disease_url = r.url
                                    break
                            except requests.RequestException:
                                continue
                        if disease_url:
                            break

                # d) inline blurb extraction from infos page if no disease page could be found
                if not disease_url:
                    for slug in cand_slugs:
                        infos_url = f"https://plantvillage.psu.edu/topics/{slug}/infos"
                        soup_infos = _fetch_html(sess, infos_url)
                        if not soup_infos:
                            continue
                        inline_desc, inline_sym, inline_cause, inline_mgmt = _extract_inline_for_disease(
                            soup_infos, disease, max_desc_sentences=max_sentences
                        )
                        if inline_desc or inline_sym or inline_cause or inline_mgmt:
                            if inline_desc:
                                updated[plant][disease]["description"] = inline_desc
                            if inline_sym:
                                updated[plant][disease]["symptoms"] = inline_sym
                            if inline_cause:
                                updated[plant][disease]["cause"] = inline_cause
                            if inline_mgmt:
                                if clean_management:
                                    # Clean up duplicate management lines when enabled
                                    updated[plant][disease]["management"] = _clean_management(
                                        inline_mgmt)
                                else:
                                    updated[plant][disease]["management"] = inline_mgmt
                            updated[plant][disease]["source"] = infos_url
                            stats["updated"] += 1
                            inline_applied = True
                            if verbose:
                                print(
                                    f"[OK][INLINE] {plant} / {disease}: {infos_url}")
                            break

            # If we successfully applied an inline blurb, skip further processing for this entry
            if inline_applied:
                if delay:
                    time.sleep(delay)
                continue

            # 3) Google CSE fallback
            if not disease_url and allow_google:
                links = _google_cse_search(plant, disease, per_page=5)
                stats["google_used"] += 1  # count attempts, not only successes
                if verbose:
                    print(
                        f"[GOOGLE] {plant} / {disease} → {len(links)} candidates")
                for link in links:
                    if "plantvillage.psu.edu" not in (link or ""):
                        continue
                    if "/diseases-and-pests/" not in link:
                        continue
                    disease_url = link
                    break

            # If we still don't have a URL and Google is allowed, try CSE fallback
            if not disease_url and allow_google:
                # already tried CSE; keep disease_url as None and mark as failed below
                pass

            if not disease_url:
                stats["failed"] += 1
                if verbose:
                    print(
                        f"[FAIL] {plant} / {disease}: could not locate disease page from {src}")
                if delay:
                    time.sleep(delay)
                continue

            soup_d = _fetch_html(sess, disease_url) if disease_url else None
            # First try scoped extraction within the specific disease section (Approach 1)
            if soup_d:
                sc_sum, sc_sym, sc_cause, sc_mgmt = _extract_scoped_sections_from_pv_page(
                    soup_d, disease, max_sentences=max_sentences
                )
                if sc_sum:
                    updated[plant][disease]["description"] = sc_sum
                if sc_sym:
                    updated[plant][disease]["symptoms"] = sc_sym
                if sc_cause:
                    updated[plant][disease]["cause"] = sc_cause
                if sc_mgmt:
                    if clean_management:
                        # Clean up duplicate management lines when enabled
                        updated[plant][disease]["management"] = _clean_management(
                            sc_mgmt)
                    else:
                        updated[plant][disease]["management"] = sc_mgmt

                updated[plant][disease]["source"] = disease_url
                stats["updated"] += 1

            # ...existing code for timing/delay and final stats...
    # ...existing code for writing updated KB JSON...
    if clean_management:
        clean_management_fields(updated)

    if list_meta:
        out_data = _apply_updates_to_rows(list_meta, updated)
    else:
        out_data = updated

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Refresh PlantVillage KB descriptions.")
    parser.add_argument("--in", dest="input", required=True,
                        help="Input KB JSON path")
    parser.add_argument(
        "--out", help="Output KB JSON path (defaults to input)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Delay between requests (seconds)")
    parser.add_argument("--max-sentences", type=int, default=2,
                        help="Max sentences for summaries (0=unlimited)")
    parser.add_argument("--only-empty", action="store_true",
                        help="Only update entries missing descriptions")
    parser.add_argument("--allow-google", action="store_true",
                        help="Enable Google CSE fallback")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing populated fields")
    parser.add_argument("--plant", help="Only process plants matching regex")
    parser.add_argument(
        "--disease", help="Only process diseases matching regex")
    parser.add_argument("--clean-management", action="store_true",
                        help="Clean duplicate management lines (disabled by default)")

    args = parser.parse_args()
    out_path = args.out or args.input

    stats = refresh_descriptions(
        args.input,
        out_path,
        args.delay,
        args.max_sentences,
        args.only_empty,
        args.allow_google,
        args.verbose,
        args.force,
        args.plant,
        args.disease,
        args.clean_management,  # pass the new flag here
    )

    print(
        f"Processed {stats['checked']} entries: {stats['updated']} updated, {stats['skipped']} skipped, {stats['failed']} failed")
    if stats['google_used'] > 0:
        print(f"Google CSE used for {stats['google_used']} lookups")
