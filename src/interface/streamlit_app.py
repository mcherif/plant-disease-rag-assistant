"""
Plant Disease RAG Assistant — Streamlit UI

This is the main interactive frontend for the plant disease RAG assistant.
Features:
- Upload plant images for disease classification (ViT finetuned model)
- Ask questions about plant diseases and management using RAG (retrieval-augmented generation)
- Uses OpenAI LLM backend and PlantVillage + Wikipedia knowledge base
- Sidebar settings: index directory, top-k context, device, detected labels
- Displays sources and context for answers
- Replaces the previous Gradio app (see app_gradio.py, now obsolete)

Usage:
- Run via Streamlit: `streamlit run src/interface/streamlit_app.py`; you may need to set PYTHONPATH first with (# Windows PowerShell): $env:PYTHONPATH = "."
- Configure index and API key via sidebar or environment variables

Author: Mohamed Cherif / innerloopinc@gmail.com
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import torch
import streamlit as st
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from src.llm.rag_pipeline import RAGPipeline, RetrievalConfig
import re
import pandas as pd
import unicodedata
import hashlib

DEBUG_MINIMAL = False  # set True to sanity-check Space/Container boot

LOGO_PATH = "images/plant-disease-rag-assistant-logo.png"

st.set_page_config(page_title="Plant Disease RAG Assistant", layout="wide")
st.sidebar.image(LOGO_PATH)
st.title("Plant Disease RAG Assistant")

st.markdown(
    """
    📖 [Read more about this app in the README](https://github.com/mcherif/Plant-Disease-RAG-Assistant/blob/main/README.md)
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #f7f9fb;
        }
        div[data-testid="stCameraInput"] {
            padding: 0.75rem;
            border: 1px solid #d1d5db;
            border-radius: 16px;
            background: #ffffff;
            max-width: 320px;
            margin-left: auto;
            margin-right: auto;
        }
        div[data-testid="stCameraInput"] video,
        div[data-testid="stCameraInput"] img {
            border-radius: 12px;
        }
        .small-help {
            font-size: 0.85rem;
            color: #6b7280;
            margin-top: 0.5rem;
        }
    </style>
    """
    ,
    unsafe_allow_html=True
)



def load_supported_crops():
    # Reads the plant names from the CSV file
    here = Path(__file__).resolve().parents[2]
    candidates = [
        here / "plant_diseases_table.csv",
        here / "data" / "plant_diseases_table.csv",
    ]
    csv_path = None
    for c in candidates:
        if c.exists():
            csv_path = c
            break
    if csv_path is None:
        raise FileNotFoundError(
            "plant_diseases_table.csv not found. Expected at project root or data/plant_diseases_table.csv"
        )
    df = pd.read_csv(csv_path)
    # Find the column name for plants (case-insensitive)
    plant_col = None
    for col in df.columns:
        if col.strip().lower() in ["plant", "crop", "species"]:
            plant_col = col
            break
    if plant_col is None:
        st.warning(
            "Could not find a column for plants/crops in plant_diseases_table.csv.")
        return []
    return [str(x).strip() for x in df[plant_col].dropna().unique()]


def plant_icon(name: str) -> str:
    icons = {
        "apple": "🍎",
        "blueberry": "🫐",
        "cherry": "🍒",
        "corn": "🌽",
        "grape": "🍇",
        "orange": "🍊",
        "peach": "🍑",
        "potato": "🥔",
        "raspberry": "🫐",
        "soybean": "🌱",
        "squash": "🥒",
        "strawberry": "🍓",
        "tomato": "🍅",
        "olive": "🫒",
    }
    # Remove 'maize' from display name for corn
    key = name.lower().replace(" (including sour)", "").replace(
        "(maize)", "").replace("maize", "").strip().split(" (")[0]
    # Force single-line display for berry icons by adding a non-breaking space after the emoji
    if key in ["blueberry", "raspberry", "strawberry"]:
        return icons.get(key, "🪴") + "\u00A0"
    if key == "corn":
        return icons.get("corn", "🪴")
    return icons.get(key, "🪴")  # Default icon


def normalize_plant_name(name):
    return name.lower().replace(" (including sour)", "").replace("(maize)", "").replace("maize", "").strip()


def render_supported_crops(highlight: str | None = None):
    crops = load_supported_crops()
    st.subheader("Supported crops")
    ncols = 14  # Display 14 plants per row
    cols = st.columns(ncols)
    for i, c in enumerate(crops):
        raw_name = str(c).strip()
        display_name = re.sub("\\s+", " ", raw_name.replace(" (including sour)", "").replace(
            "(maize)", "").replace("maize", "").replace(", bell", "").strip())
        icon = plant_icon(display_name)
        is_hit = bool(highlight) and normalize_plant_name(raw_name) == normalize_plant_name(str(highlight))
        card_css = f"""
            <div style="
                padding:8px 8px;
                border-radius:14px;
                border:2px solid {'#22c55e' if is_hit else '#e5e7eb'};
                background:{'#ecfdf5' if is_hit else '#ffffff'};
                text-align:center;
                font-weight:{'700' if is_hit else '500'};
                margin-bottom:8px;
                font-size:1em;
            ">{icon} {display_name}</div>
        """
        with cols[i % ncols]:
            st.markdown(card_css, unsafe_allow_html=True)


# Place this immediately after st.title and before any other widget
if "rag_valid" not in st.session_state:
    st.session_state["rag_valid"] = True

current_detected = st.session_state.get("detected_plant")
render_supported_crops(current_detected)

# Sidebar config (move this block up, before any function that uses MODEL_DIR)
st.sidebar.header("Settings")
MODEL_DIR = st.sidebar.text_input("Model directory", "models/vit-finetuned")
index_dir = st.sidebar.text_input("Index dir", "models/index/kb-faiss-bge")
top_k = st.sidebar.slider("Top-k context", 1, 6, 3)
# Only show CUDA as an option if it's actually available
available_devices = ["cpu"]
if torch.cuda.is_available():
    available_devices.append("cuda")
retrieval_device = st.sidebar.selectbox("Device", available_devices, index=0)
model_env = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
st.sidebar.caption(f"Judge/LLM model (env): {model_env}")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.warning("OPENAI_API_KEY not set. Answers won’t work.", icon="⚠️")

show_dashboard = st.sidebar.checkbox("Show Dashboard")
if show_dashboard:
    st.header("Feedback Dashboard")
    try:
        df = pd.read_json("data/feedback/feedback.jsonl", lines=True)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], errors="coerce", utc=True, format=None
            )
            df = df.dropna(subset=["timestamp"])
        if df.empty:
            st.info("No feedback data yet.")
        else:
            st.bar_chart(df["satisfaction"].value_counts())
            st.line_chart(df.groupby(df["timestamp"].dt.date).size())
            st.write("Recent feedback:", df.tail(10))
    except Exception as e:
        st.warning(f"No feedback data yet or error loading dashboard: {e}")

DEBUG_MODE = st.sidebar.checkbox("Show debug info", value=False)

# ---- helpers (must be defined before use) ----


def _canon_plant(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    aliases = {
        "peach tree": "Peach",
        "maize": "Corn (maize)",
        "corn": "Corn (maize)",
        "pepper": "Pepper, bell",
        "bell pepper": "Pepper, bell",
    }
    key = s.lower()
    return aliases.get(key, s.title())


def _canon_disease(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    s = re.sub(r"^[\s,;:-]+", "", s)
    # Remove parenthetical qualifiers like (maize)
    s = re.sub(r"\(.*?\)", "", s)
    s = s.replace("  ", " ").strip()
    aliases = {
        # common classifier labels -> KB names
        "peach bacterial": "Bacterial spot",
        "bacterial spot of peach": "Bacterial spot",
        "powdery mildew": "Powdery mildew",
        "northern leaf blight": "Northern Leaf Blight",
        "gray leaf spot": "Cercospora leaf spot Gray leaf spot",
        "common rust": "Common rust",
        "bacterial spot": "Bacterial spot",
        "bell bacterial spot": "Bacterial spot",
        "pepper bell bacterial spot": "Bacterial spot",
        "bell pepper bacterial spot": "Bacterial spot",
        "pepper bacterial spot": "Bacterial spot",
    }
    key = s.lower()
    if key.endswith(" healthy") or key == "healthy":
        return "healthy"
    return aliases.get(key, re.sub(r"\s+", " ", s).strip())


# Debug/test: show canonicalization result in debug mode
if DEBUG_MODE:
    st.write("_canon_disease('Apple scab') =", _canon_disease(
        "Apple scab"))  # Should return "Apple scab"


def _infer_labels_from_classifier(raw: str):
    """
    Heuristic: infer (plant, disease) from a raw classifier label like 'peach bacterial'.
    Uses canonicalizers and simple keyword detection.
    """
    if not raw:
        return None, None
    s = re.sub(r"[_\-]+", " ", str(raw)).strip()
    low = s.lower()
    plant_keys = ["peach", "tomato", "potato", "apple", "grape", "corn", "maize",
                  "pepper", "orange", "banana", "cucumber", "zucchini", "strawberry", "raspberry", "soybean", "squash", "cherry", "olive"]
    plant = None
    matched_key = None
    for k in plant_keys:
        if k in low:
            plant = _canon_plant(k)
            matched_key = k
            break
    disease_raw = s
    if matched_key:
        disease_raw = re.sub(
            rf"\b{re.escape(matched_key)}\b", " ", disease_raw, flags=re.I)
        plant_alias = _canon_plant(matched_key) or matched_key
        plant_tokens = re.findall(r"\w+", plant_alias.lower())
        for token in plant_tokens:
            disease_raw = re.sub(
                rf"\b{re.escape(token)}\b", " ", disease_raw, flags=re.I)
        disease_raw = re.sub(r"\s+", " ", disease_raw)
        disease_raw = disease_raw.strip().lstrip(",;:- ")
    disease = _canon_disease(disease_raw.strip())
    if not disease or disease.lower() == (plant or "").lower():
        disease = _canon_disease(s)
    # Fix: If disease is a generic term and plant is present, prepend plant
    generic_diseases = {"scab", "rust", "blight",
                        "spot", "mildew", "rot", "smut"}
    if (
        disease
        and plant
        and " " not in disease
        and disease.lower() in generic_diseases
        and not disease.lower().startswith(plant.lower())
    ):
        plant_prefix = plant.replace(",", " ")
        plant_prefix = re.sub(r"\s+", " ", plant_prefix).strip()
        disease = f"{plant_prefix} {disease}".strip()
    return plant, disease or None


def normalize(text):
    # Lowercase, remove accents, strip whitespace
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    return text.strip()


def extract_main_disease_name(title: str) -> str:
    """
    Extracts the main disease name from a KB title like 'Apple scab — Venturia inaequalis'
    or 'Apple scab (Venturia inaequalis)'.
    """
    # Split on em dash, hyphen, or parenthesis
    title = re.split(r"[—\-–(]", title)[0]
    return normalize(title)


# Debug/test: normalize and compare
if DEBUG_MODE:
    query = 'apple scab'
    kb_entry = "Apple scab — Venturia inaequalis"
    # or use '-' if that's the delimiter
    main_name = kb_entry.split('—')[0].strip()
    if normalize(query) == normalize(main_name):
        st.write("Match!")
    else:
        st.write("No match.")

# ---- helpers end ----


def load_model_and_processor():
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR, use_fast=True)
    device = torch.device(retrieval_device)
    # Force loading with actual weights (not meta tensors)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_DIR,
        low_cpu_mem_usage=False,  # Disable lazy loading
        torch_dtype=torch.float32
    )
    model = model.to(device)
    return model, processor, device


def id2label():
    # Load mapping from class_mapping.json
    with open(f"{MODEL_DIR}/class_mapping.json", "r") as f:
        class_map = json.load(f)
    # Invert mapping: {index: class_name}
    mapping = {v: k for k, v in class_map.items()}
    if DEBUG_MODE:
        st.write("[DEBUG] id2label mapping sample:",
                 dict(list(mapping.items())[:10]))
    return mapping

# ---- Supported crops section (always visible; highlights after upload) ----


IMAGE_SOURCE_UPLOAD = "Upload from file"
IMAGE_SOURCE_CAMERA = "Use camera"

st.subheader("Diagnose from a photo")
st.caption("Upload or snap a plant photo to get instant suggestions from the classifier.")

controls_col, preview_col = st.columns([1, 1.1])
image_file = None
with controls_col:
    st.markdown("**Add a photo**")
    image_source = st.radio(
        "Image source",
        (IMAGE_SOURCE_UPLOAD, IMAGE_SOURCE_CAMERA),
        horizontal=True,
        key="image_source_selector",
    )
    if image_source == IMAGE_SOURCE_UPLOAD:
        image_file = st.file_uploader(
            "Upload a plant image", type=["jpg", "jpeg", "png"], key="plant_image_file",
            help="Use a clear photo showing leaves or fruit.")
    else:
        image_file = st.camera_input(
            "Capture a plant photo", key="plant_camera_capture",
            help="Allow access to your camera for a quick snapshot.")
        st.markdown(
            '<p class="small-help">Tip: natural light and sharp focus improve results.</p>',
            unsafe_allow_html=True,
        )

image = None
image_caption = ""
if image_file is not None:
    raw_bytes = image_file.getvalue()
    image_hash = hashlib.md5(raw_bytes).hexdigest() if raw_bytes else None
    if image_hash and st.session_state.get("last_image_hash") != image_hash:
        st.session_state["last_image_hash"] = image_hash
        st.session_state["rag_valid"] = False
    if hasattr(image_file, "seek"):
        image_file.seek(0)
    if DEBUG_MODE:
        name = getattr(image_file, "name", "camera_capture")
        st.caption(f"Input source: {image_source}, name: {name}")
    image = Image.open(image_file).convert("RGB")
    image_caption = "Captured image" if image_source == IMAGE_SOURCE_CAMERA else "Uploaded image"

with preview_col:
    if image is not None:
        st.image(image, caption=image_caption, width=260)


if image is not None:
    with st.spinner("Loading model..."):
        model, processor, model_device = load_model_and_processor()
    labels = id2label()

    with st.spinner("Running inference..."):
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()
            topk = min(3, probs.shape[-1])
            scores, idxs = torch.topk(probs, topk)

    st.subheader("Top predictions")
    for score, idx in zip(scores.tolist(), idxs.tolist()):
        if DEBUG_MODE:
            st.write(
                f"[DEBUG] idx={idx}, type={type(idx)}, label_name={labels.get(idx)}")
        label_name = labels.get(idx, f"LABEL_{idx}")
        plant, disease = _infer_labels_from_classifier(label_name)
        if disease:
            st.write(f"- {label_name}: {score:.3f} ({disease})")
        else:
            st.write(f"- {label_name}: {score:.3f}")

    # Take top-1 as detected labels and store to session
    top1_label = labels.get(int(idxs[0]), f"class_{int(idxs[0])}")
    plant_guess, disease_guess = _infer_labels_from_classifier(top1_label)
    if (
        st.session_state.get("detected_plant") != plant_guess
        or st.session_state.get("detected_disease") != disease_guess
    ):
        st.session_state["detected_plant"] = plant_guess
        st.session_state["detected_disease"] = disease_guess
        st.session_state["plant_input"] = plant_guess or ""
        st.session_state["disease_input"] = disease_guess or ""
        st.session_state["rag_valid"] = False
        st.session_state["force_rerun"] = not st.session_state.get(
            "force_rerun", False)
        st.rerun()
        st.stop()  # Prevent further code from running in this pass

# RAG pipeline (re-init when settings change)
cfg = RetrievalConfig(index_dir=index_dir, top_k=top_k,
                      device=retrieval_device)
rag = RAGPipeline(cfg)

# Detected labels from classifier
detected_plant = st.session_state.get("detected_plant")
detected_disease = st.session_state.get("detected_disease")

# Inputs
use_labels = st.checkbox(
    "Use detected plant/disease (if available)", value=True)
# Set defaults if keys missing
if "plant_input" not in st.session_state:
    st.session_state["plant_input"] = (detected_plant or "")
if "disease_input" not in st.session_state:
    st.session_state["disease_input"] = (detected_disease or "")
# If user opted-in and fields are empty, copy detected labels now
if use_labels and (detected_plant or detected_disease):
    if detected_plant and not (st.session_state.get("plant_input") or "").strip():
        st.session_state["plant_input"] = detected_plant
    if detected_disease and not (st.session_state.get("disease_input") or "").strip():
        st.session_state["disease_input"] = detected_disease
    st.caption(
        f"Detected: plant={detected_plant or '-'} | disease={detected_disease or '-'}"
    )

col1, col2 = st.columns(2)
# Now create the text inputs (they will read session_state defaults)
with col1:
    plant = st.text_input("Plant (optional)",
                          key="plant_input", placeholder="Peach")
with col2:
    disease = st.text_input("Disease (optional)",
                            key="disease_input", placeholder="Bacterial spot")

query = st.text_area(
    "Question", placeholder="e.g., What can I do to treat this?")

if DEBUG_MODE:
    st.write(f"[DEBUG] Plant: {plant}, Disease: {disease}")

run = st.button("Get answer", type="primary")

# Run
DEFAULT_QUESTION = "What can I do to treat this?"
if run:
    q = (query or "").strip() or DEFAULT_QUESTION
    plant_norm = _canon_plant(st.session_state.get("plant_input") or "")
    disease_norm = _canon_disease(st.session_state.get("disease_input") or "")
    plant_norm = plant_norm or None
    disease_norm = disease_norm or None

    # Always include both main and full disease names if available and different
    disease_main = extract_main_disease_name(
        disease_norm) if disease_norm else None
    disease_full = disease_norm
    disease_names = []
    if disease_main:
        disease_names.append(disease_main)
    if disease_full and disease_full != disease_main:
        disease_names.append(disease_full)
    # Remove duplicates while preserving order
    disease_names = list(dict.fromkeys([d for d in disease_names if d]))

    if use_labels and (plant_norm or disease_names):
        # Prefer a natural language question
        disease_part = disease_names[0] if disease_names else ""
        if plant_norm and disease_part:
            q = f"How do I treat {disease_part} on {plant_norm}?"
        elif plant_norm:
            q = f"What diseases affect {plant_norm} and how can I treat them?"
        elif disease_part:
            q = f"How do I treat {disease_part}?"
        # Optionally, append the user's question if it's not the default
        if query and query.strip() and query.strip() != DEFAULT_QUESTION:
            q += f" {query.strip()}"

    if DEBUG_MODE:
        st.write(
            f"[DEBUG] RAG query: {q}, plant: {plant_norm}, disease: {disease_norm}")

    with st.spinner("Retrieving and generating..."):
        try:
            res = rag.answer(q, plant=plant_norm, disease=disease_norm)
            # Debug: Show retrieved KB titles
            if DEBUG_MODE:
                st.write("[DEBUG] Retrieved KB titles:")
                for doc in res.get("retrieved", []):
                    meta = doc.get("meta", {})
                    st.write(meta.get("title") or meta.get("doc_id"))
        except Exception as e:
            st.error(f"RAG error: {e}")
        else:
            st.session_state["last_answer"] = res.get("answer", "")
            st.session_state["last_question"] = q
            st.session_state["last_plant"] = plant_norm
            st.session_state["last_disease"] = disease_norm
            st.session_state["last_sources"] = res.get("retrieved", [])
            st.session_state["feedback_submitted"] = False
            st.session_state["rag_valid"] = True

# Always show the answer and feedback form if available
if st.session_state.get("last_answer"):
    rag_valid = st.session_state.get("rag_valid", True)
    if not rag_valid:
        st.subheader("Answer")
        st.info("Detected a new image. Run 'Get answer' to refresh this response.")
        with st.expander("Previous answer (stale)", expanded=False):
            st.write(st.session_state["last_answer"])
            retrieved = st.session_state.get("last_sources", []) or []
            if retrieved:
                for i, doc in enumerate(retrieved, start=1):
                    meta = doc.get("meta", {})
                    title = meta.get("title") or meta.get("doc_id") or f"Doc {i}"
                    url = meta.get("url")
                    bullet = f"[{i}] {title}"
                    if url:
                        st.markdown(f"{bullet} - [{url}]({url})")
                    else:
                        st.write(bullet)
            else:
                st.info("No sources were retrieved for the previous answer.")
    else:
        st.subheader("Answer")
        st.write(st.session_state["last_answer"])
        retrieved = st.session_state.get("last_sources", []) or []
        if retrieved:
            st.subheader("Sources")
            for i, doc in enumerate(retrieved, start=1):
                meta = doc.get("meta", {})
                title = meta.get("title") or meta.get("doc_id") or f"Doc {i}"
                url = meta.get("url")
                with st.expander(f"[{i}] {title}"):
                    if url:
                        st.markdown(f"[{url}]({url})")
                    st.write(meta.get("text", "")[:1200])
        else:
            st.info("No sources retrieved.")

        if "feedback_submitted" not in st.session_state:
            st.session_state["feedback_submitted"] = False

        if not st.session_state["feedback_submitted"]:
            st.subheader("Your Feedback")
            col_fb1, col_fb2 = st.columns([1, 4])
            with col_fb1:
                satisfaction = st.radio(
                    "Was this answer helpful?",
                    ["👍 Yes", "👎 No"],
                    horizontal=True,
                    key="satisfaction_radio"
                )
            with col_fb2:
                user_comment = st.text_area(
                    "Additional comments (optional)",
                    key="user_comment_area"
                )
            if st.button("Submit Feedback", key="submit_feedback_btn"):
                import datetime
                feedback_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "question": st.session_state["last_question"],
                    "plant": st.session_state["last_plant"],
                    "disease": st.session_state["last_disease"],
                    "answer": st.session_state["last_answer"],
                    "satisfaction": satisfaction,
                    "comment": user_comment
                }
                os.makedirs("data/feedback", exist_ok=True)
                with open("data/feedback/feedback.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(feedback_entry, ensure_ascii=False) + "\n")
                st.session_state["feedback_submitted"] = True
                st.success("Thank you for your feedback!")
        else:
            st.info("Feedback already submitted. Thank you!")

