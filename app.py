import numpy as np
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader

# ============================
# CONFIG
# ============================
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-large"   # improved generation quality
TOP_K = 5


# ============================
# PDF & TEXT UTILITIES
# ============================
def extract_text_from_pdf_filelike(file) -> str:
    reader = PdfReader(file)
    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            pages_text.append("")
    return "\n".join(pages_text)


def build_chunks(text: str, chunk_size: int = 800, overlap: int = 150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ============================
# LOAD MODELS
# ============================
@st.cache_resource(show_spinner="Loading Science Models...")
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME).to(device)
    return device, embedder, tokenizer, gen_model


# ============================
# INDEX UTILITIES
# ============================
def build_index_from_files(uploaded_files):
    docs = []
    for f in uploaded_files:
        text = extract_text_from_pdf_filelike(f)
        chunks = build_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for idx, c in enumerate(chunks):
            docs.append({"doc_id": f.name, "chunk_id": idx, "text": c})
    return docs


def build_faiss_index(docs, embedder):
    dim = embedder.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dim)

    vecs = []
    for d in docs:
        v = embedder.encode(d["text"], convert_to_numpy=True)
        vecs.append(v)

    vecs = np.vstack(vecs).astype("float32")
    index.add(vecs)
    return index


def retrieve_context(query, index, embedder, docs):
    q_vec = embedder.encode(query, convert_to_numpy=True).astype("float32")
    q_vec = np.expand_dims(q_vec, axis=0)
    scores, indices = index.search(q_vec, TOP_K)
    return [docs[i] for i in indices[0]]


# ============================
# PROMPT + GENERATION
# ============================
def build_prompt(topic, chunks, target_class, num_questions):
    context = "\n\n".join([c["text"] for c in chunks])

    prompt = f"""
You are an NCERT Science teacher for Class {target_class}.
Generate {num_questions} exam-style LONG ANSWER QUESTIONS ONLY
based on the topic: "{topic}"

Rules:
- DO NOT give answers. Only give questions.
- Questions should start with: Explain, Describe, What do you mean by, How does, Why, etc.
- Questions must be detailed and require at least 4–8 line answers.
- Use only the textbook context given below.

CONTEXT:
{context}

Now generate the questions:
"""
    return prompt


def generate_questions(topic, target_class, num_questions, index, docs, embedder, tokenizer, model, device):
    chunks = retrieve_context(topic, index, embedder, docs)
    prompt = build_prompt(topic, chunks, target_class, num_questions)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=350,
            num_beams=5,
            temperature=0.7,
            no_repeat_ngram_size=3
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text, chunks


# ============================
# STREAMLIT UI
# ============================
def main():
    st.set_page_config(page_title="NCERT Science Question Generator", layout="wide")
    st.title("🔬 NCERT Science – Long Answer Question Generator (6–10)")

    uploaded = st.file_uploader("Upload NCERT Science PDFs (Class 6–10)", type=["pdf"], accept_multiple_files=True)

    class_choice = st.selectbox("Select Class", [6, 7, 8, 9, 10], index=2)
    num_q = st.slider("How many questions?", 3, 10, 5)
    topic = st.text_input("Enter Topic (Example: Nutrition in Plants, Acids Bases Salts, Motion, Electricity)")

    if not uploaded:
        st.info("Please upload at least one NCERT Science PDF.")
        return

    device, embedder, tokenizer, gen_model = load_models()

    with st.spinner("Processing PDFs and building index..."):
        docs = build_index_from_files(uploaded)
        index = build_faiss_index(docs, embedder)

    if topic and st.button("Generate Questions"):
        with st.spinner("Generating questions..."):
            questions, used_chunks = generate_questions(
                topic, class_choice, num_q, index, docs, embedder, tokenizer, gen_model, device
            )

        st.subheader("📄 Generated Science Questions")
        st.write(questions)

        with st.expander("Show textbook chunks used"):
            for c in used_chunks:
                st.markdown(f"**{c['doc_id']} – chunk {c['chunk_id']}**")
                st.write(c["text"])
                st.markdown("---")


if __name__ == "__main__":
    main()
