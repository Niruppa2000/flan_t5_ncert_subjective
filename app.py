# app.py
import os
import re
import json
import math
import random
import numpy as np
import streamlit as st
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader

# ============================
# CONFIG (tweak as necessary)
# ============================
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"  # change to -small for lower RAM
TOP_K = 6  # number of chunks to retrieve
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# UTILITIES: text extraction & chapter detection
# ============================
def extract_pages_text(file) -> List[str]:
    """Return list of page texts for a file-like (streamlit upload)."""
    reader = PdfReader(file)
    pages = []
    for p in reader.pages:
        try:
            pages.append((p.extract_text() or "").strip())
        except Exception:
            pages.append("")
    return pages

def find_chapter_boundaries(page_texts: List[str]) -> List[Tuple[int, str]]:
    """
    Heuristically find chapter start pages and titles.
    Returns list of (page_index, chapter_title).
    Strategy:
    - look for lines with 'CHAPTER' or 'Chapter' or 'C H A P T E R'
    - capture nearby lines (same page) as title if present
    - fallback: look for 'Chapter X' patterns in the page text
    """
    starts = []
    for i, p in enumerate(page_texts):
        txt = p.upper()
        # quick check for 'CHAPTER' presence
        if "CHAPTER" in txt or "C H A P T E R" in txt:
            # try to extract a title line nearby
            # split original page into lines (preserve case for title)
            lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
            # find line index containing 'chapter'
            found_idx = None
            for li, ln in enumerate(lines):
                if re.search(r"\bCHAPTER\b", ln.upper()) or re.search(r"\bC H A P T E R\b", ln.upper()):
                    found_idx = li
                    break
            # candidate title is next non-empty line after 'CHAPTER' line
            title = None
            if found_idx is not None:
                # look next 1-3 lines for a title-like string (not too long)
                for j in range(found_idx+1, min(found_idx+4, len(lines))):
                    candidate = lines[j]
                    if len(candidate) > 3 and len(candidate) < 120:
                        title = candidate
                        break
            # fallback: try regex "CHAPTER X: Title" on upper-case txt
            if title is None:
                m = re.search(r"CHAPTER\s*\d+\s*[:.\-]?\s*(.+)", p, flags=re.IGNORECASE)
                if m:
                    title = m.group(1).strip().split("\n")[0]
            # final fallback: use first long line on page as title
            if title is None:
                for ln in lines[:6]:
                    if len(ln) > 5 and len(ln) < 120:
                        title = ln
                        break
            if title is None:
                title = f"Chapter_on_page_{i+1}"
            starts.append((i, title.strip()))
    # Ensure sorted and unique page starts
    starts_sorted = sorted(starts, key=lambda x: x[0])
    # remove very close duplicates
    filtered = []
    last_page = -10
    for p, t in starts_sorted:
        if p - last_page > 1:
            filtered.append((p, t))
            last_page = p
    return filtered

def build_chapter_map_for_file(file) -> Dict[str, Dict]:
    """
    For one uploaded PDF, produce a mapping:
    {
      "chapter_title_normalized": {
          "title": original_title,
          "start_page": int,
          "end_page": int,
          "text": "concatenated chapter text",
          "chunks": [ { "chunk_id": 0, "text": "...", "page": p } , ... ]
      }, ...
    }
    """
    pages = extract_pages_text(file)
    # find chapter starts
    starts = find_chapter_boundaries(pages)
    chapters = {}

    if not starts:
        # whole document as one pseudo-chapter
        full = "\n".join(pages)
        normalized = "full_document"
        chapters[normalized] = {
            "title": "Full Document",
            "start_page": 0,
            "end_page": len(pages)-1,
            "text": full,
            "chunks": []
        }
    else:
        # infer end pages from next start
        for idx, (start_page, title) in enumerate(starts):
            end_page = (starts[idx+1][0] - 1) if idx+1 < len(starts) else (len(pages)-1)
            chapter_text = "\n".join(pages[start_page:end_page+1])
            normalized = re.sub(r'\s+', ' ', title).strip().lower()
            chapters[normalized] = {
                "title": title.strip(),
                "start_page": start_page,
                "end_page": end_page,
                "text": chapter_text,
                "chunks": []
            }

    # chunk chapter texts
    for norm, meta in chapters.items():
        text = meta["text"]
        # simple chunking by characters keeping overlap
        chunks = []
        s = 0
        c_id = 0
        while s < len(text):
            e = min(len(text), s + CHUNK_SIZE)
            chunk = text[s:e].strip()
            # try to not cut mid-sentence (look back)
            if len(chunk) > 200:
                last_dot = max(chunk.rfind("."), chunk.rfind("।"))
                if last_dot != -1 and e != len(text):
                    # keep up to last_dot
                    e = s + last_dot + 1
                    chunk = text[s:e].strip()
            if len(chunk) > 80:
                chunks.append({"chunk_id": c_id, "text": chunk})
                c_id += 1
            s = max(0, e - CHUNK_OVERLAP)
        meta["chunks"] = chunks

    return chapters

# ============================
# EMBEDDING & GENERATION MODELS (cached)
# ============================
@st.cache_resource(show_spinner="Loading models (embedding + Flan-T5)...")
def load_models():
    device = DEVICE
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME).to(device)
    gen_model.eval()
    return device, embedder, tokenizer, gen_model

# ============================
# FAISS INDEX (chapter-level)
# ============================
def build_docs_and_index(uploaded_files) -> Tuple[List[Dict], faiss.IndexFlatL2]:
    """
    Build doc list: each doc is a chunk with metadata:
    { "file_name", "chapter_norm", "chapter_title", "chunk_id", "text" }
    and build FAISS index for all chunk embeddings.
    """
    device, embedder, _, _ = load_models()
    docs = []
    for f in uploaded_files:
        chapters = build_chapter_map_for_file(f)
        fname = getattr(f, "name", "uploaded_pdf")
        for c_norm, meta in chapters.items():
            for ch in meta["chunks"]:
                docs.append({
                    "file_name": fname,
                    "chapter_norm": c_norm,
                    "chapter_title": meta["title"],
                    "chunk_id": ch["chunk_id"],
                    "text": ch["text"]
                })

    if not docs:
        return docs, None

    emb_dim = embedder.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(emb_dim)

    # embed in small batches to save memory
    batch_size = 16
    vectors = []
    for i in range(0, len(docs), batch_size):
        texts = [d["text"] for d in docs[i:i+batch_size]]
        vecs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        vectors.append(vecs.astype("float32"))
    all_vecs = np.vstack(vectors)
    index.add(all_vecs)
    return docs, index

# ============================
# RETRIEVAL (chapter-scoped)
# ============================
def retrieve_chunks_for_chapter(chapter_norm: str, docs: List[Dict], index: faiss.IndexFlatL2, top_k=TOP_K) -> List[Dict]:
    """
    Restrict search to chunks belonging to chapter_norm by:
    - filter docs for that chapter, build local index for them and search
    This ensures we only retrieve chunks from the requested chapter.
    """
    chapter_docs = [d for d in docs if d["chapter_norm"] == chapter_norm]
    if not chapter_docs:
        return []

    # embed query by taking a short automatic query as context: use chapter title
    # but to keep consistent we can also build index of chapter_docs only and perform a similarity on the chapter text
    # Here we'll build small local index (cheap because chapters are small)
    emb_dim = index.d
    local_index = faiss.IndexFlatL2(emb_dim)
    # we need the embedding model again
    device, embedder, _, _ = load_models()
    vectors = embedder.encode([d["text"] for d in chapter_docs], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    local_index.add(vectors)

    # we will use the chapter title as the query vector (short) to pick representative chunks
    q_text = chapter_docs[0]["chapter_title"] if chapter_docs else ""
    q_vec = embedder.encode([q_text], convert_to_numpy=True).astype("float32")
    distances, indices = local_index.search(q_vec, min(top_k, len(chapter_docs)))
    indices = indices[0]
    retrieved = [chapter_docs[i] for i in indices]
    return retrieved

# ============================
# CLEAN & POST-PROCESS MODEL OUTPUT
# ============================
def clean_and_extract_questions(raw_text: str, chapter_title: str, num_questions: int):
    """
    Similar cleaning as before but biased to produce long-answer style questions.
    Accepts numbered lists or newline-separated outputs.
    """
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    questions = []

    for ln in lines:
        text = ln
        # remove leading numbering like "1. " or "1)"
        m = re.match(r"^\s*\d+\s*[\.\)]\s*(.*)$", text)
        if m:
            text = m.group(1).strip()
        # remove bullets
        if text.startswith("- "):
            text = text[2:].strip()
        # ignore super short junk
        if len(text.split()) < 5:
            continue
        # ensure ends with ?
        if not text.endswith("?"):
            text = text.rstrip(". ") + "?"
        questions.append(text)

    # if not enough, create teacher-style long-answer templates from chapter title
    if len(questions) < num_questions:
        templates = [
            f"Explain with examples the key points of {chapter_title}.",
            f"Describe the main ideas discussed in the chapter '{chapter_title}' and explain why they are important.",
            f"Explain any one process or example from the chapter '{chapter_title}' in detail.",
            f"What are the important definitions and terms mentioned in '{chapter_title}'? Explain them with examples.",
            f"Discuss the real-life applications or significance of the concepts explained in '{chapter_title}'."
        ]
        for t in templates:
            if len(questions) >= num_questions:
                break
            if t not in questions:
                questions.append(t)

    return questions[:num_questions]

# ============================
# GENERATION: teacher-style prompts
# ============================
def generate_questions_from_chapter(chapter_norm: str, docs, index, tokenizer, gen_model, device, num_questions=5, target_class=7):
    """
    Retrieve chapter-specific chunks, create a prompt that instructs the model to output
    teacher-style long-answer subjective questions, and return the cleaned question list plus the retrieved chunks used.
    """
    retrieved = retrieve_chunks_for_chapter(chapter_norm, docs, index, top_k=TOP_K)
    if not retrieved:
        return "", []

    context_text = "\n\n".join([f"Source: {r['file_name']} (chunk {r['chunk_id']})\n{r['text']}" for r in retrieved])

    prompt = f"""
You are an experienced NCERT science teacher for Class {target_class}.
Using ONLY the textbook passages given in CONTEXT, write {num_questions} high-quality exam-style subjective questions
(long-answer / 3-5 marks) for students of Class {target_class} on the chapter titled: "{docs[0]['chapter_title'] if docs else ''}".

Guidelines:
- Questions must be clear, teacher-style long-answer questions (Explain / Describe / Why / How / Discuss).
- Avoid one-word or yes/no questions.
- Each question must be complete and end with a question mark (?).
- Output exactly a numbered list (1., 2., 3., ...), nothing else.

CONTEXT:
{context_text}
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outs = gen_model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            do_sample=False,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    raw = tokenizer.decode(outs[0], skip_special_tokens=True).strip()
    questions = clean_and_extract_questions(raw, retrieved[0]["chapter_title"] if retrieved else "", num_questions)
    # We also return the retrieved chunks (for display)
    return "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)]), retrieved

# ============================
# STREAMLIT UI
# ============================
def main():
    st.set_page_config(page_title="NCERT Chapter → Teacher-style Questions", layout="wide")
    st.title("📘 NCERT Chapter → Teacher-style Subjective Questions (Classes 6–10)")

    st.markdown(
        """
Upload NCERT PDFs (Science) and then type / select a **chapter name** from the detected chapters (or type it manually).
The app will generate **teacher-style long-answer subjective questions** for that chapter suitable for exams.
"""
    )

    uploaded_files = st.file_uploader(
        "Upload NCERT PDFs (you can select multiple files)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        target_class = st.selectbox("Select Class", [6,7,8,9,10], index=0)
    with col2:
        num_questions = st.slider("How many questions?", 1, 8, 5)
    with col3:
        model_choice = st.selectbox("Generation model", [GEN_MODEL_NAME, "google/flan-t5-small"], index=0)

    # topic replaced by chapter_input
    chapter_input = st.text_input("Enter or select a chapter name (type a few words):", value="")

    if not uploaded_files:
        st.info("Please upload one or more NCERT PDFs first.")
        return

    # load models
    with st.spinner("Loading models (embedding + generation)..."):
        device, embedder, tokenizer, gen_model = load_models()

    # build docs & index
    with st.spinner("Reading PDFs, detecting chapters, and building index..."):
        docs, index = build_docs_and_index(uploaded_files)
        if not docs:
            st.error("No textual content could be extracted from the uploaded PDFs. (Scanned PDFs may need OCR.)")
            return

    # show detected chapters for user selection
    # build mapping: chapter_norm -> display title (first occurrence)
    chapter_map = {}
    for d in docs:
        chapter_map.setdefault(d["chapter_norm"], d["chapter_title"])

    detected_titles = list({v for k,v in chapter_map.items()})
    # small helper: fuzzy-like search for suggestions
    suggestions = []
    if chapter_input:
        q = chapter_input.strip().lower()
        for k, title in chapter_map.items():
            if q in title.lower():
                suggestions.append((k, title))
    else:
        # show first few detected chapters as suggestions
        suggestions = list(chapter_map.items())[:15]

    st.markdown("**Detected chapter titles (from uploaded PDFs)** — click to copy a title into the input box for convenience:")
    cols = st.columns(3)
    i = 0
    for k, t in list(chapter_map.items())[:30]:
        if cols[i%3].button(t):
            # set the text input (Streamlit workaround: use st.session_state)
            st.session_state["chapter_input"] = t
        i += 1

    # Update chapter_input from session state if button pressed
    if "chapter_input" in st.session_state and st.session_state["chapter_input"]:
        chapter_input = st.session_state["chapter_input"]

    # When user clicks generate
    if st.button("Generate Questions from Chapter"):
        if not chapter_input:
            st.error("Please type or select a chapter name (exact or partial match).")
            return

        # find best matching normalized chapter key
        q = chapter_input.strip().lower()
        matched_key = None
        matched_title = None
        # exact match attempt
        for k, t in chapter_map.items():
            if q == t.lower() or q == k:
                matched_key = k
                matched_title = t
                break
        # partial match attempt
        if matched_key is None:
            for k, t in chapter_map.items():
                if q in t.lower() or q in k:
                    matched_key = k
                    matched_title = t
                    break
        # fuzzy fallback: choose first where most words match
        if matched_key is None:
            qwords = set(q.split())
            best_score = 0
            for k,t in chapter_map.items():
                score = len(qwords.intersection(set(t.lower().split())))
                if score > best_score:
                    best_score = score
                    matched_key = k
                    matched_title = t

        if matched_key is None:
            st.error("Could not match the chapter name. Try a shorter/unique phrase from the chapter title.")
            return

        st.info(f"Generating {num_questions} questions for chapter: **{matched_title}**")
        # allow model selection (small or base)
        if model_choice != GEN_MODEL_NAME:
            # reload selected small model
            tokenizer = AutoTokenizer.from_pretrained(model_choice)
            gen_model = AutoModelForSeq2SeqLM.from_pretrained(model_choice).to(device)

        with st.spinner("Retrieving chapter text and generating questions... (this may take a few seconds)"):
            try:
                questions_block, retrieved = generate_questions_from_chapter(
                    chapter_norm=matched_key,
                    docs=docs,
                    index=index,
                    tokenizer=tokenizer,
                    gen_model=gen_model,
                    device=device,
                    num_questions=num_questions,
                    target_class=target_class
                )
            except Exception as e:
                st.error(f"Generation failed: {e}")
                return

        st.subheader("📄 Generated Questions (Teacher-style, long-answer)")
        st.code(questions_block)

        with st.expander("Show textbook chunks used for generation"):
            for r in retrieved:
                st.markdown(f"**{r['file_name']} — chunk {r['chunk_id']}**")
                st.write(r["text"])
                st.markdown("---")

        # optional: give user a way to download as .txt
        if st.button("Download questions as TXT"):
            bname = matched_title.replace(" ", "_")[:50]
            txt = questions_block
            st.download_button(label="Download TXT", data=txt, file_name=f"{bname}_questions.txt", mime="text/plain")

if __name__ == "__main__":
    main()
