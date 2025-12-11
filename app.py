# app.py (robust tokenizer + LoRA loader for Streamlit)
import os
import streamlit as st
import torch
import glob
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ---------- CONFIG ----------
BASE_MODEL = "google/flan-t5-base"   # change only if your LoRA used a different base
LORA_PATH = "./lora_ncert_finetuned"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="NCERT Q Generator (LoRA)", layout="wide")
st.title("NCERT Teacher-Style Question Generator — (LoRA loader debug)")

st.write("Device:", DEVICE)
st.write(f"Expecting LoRA adapter at repository path: `{LORA_PATH}`")

# Show files present in adapter folder to help debug tokeniser issues
if os.path.isdir(LORA_PATH):
    st.write("Files in adapter folder (top-level):")
    try:
        files = sorted(os.listdir(LORA_PATH))
        st.write(files)
        # show nested entries (one level)
        nested = {d: sorted(os.listdir(os.path.join(LORA_PATH, d))) for d in files if os.path.isdir(os.path.join(LORA_PATH, d))}
        if nested:
            st.write("Nested folders (one level):")
            st.write(nested)
    except Exception as e:
        st.warning(f"Could not list adapter files: {e}")
else:
    st.error(f"Adapter folder not found at `{LORA_PATH}`. Please upload it to the repo root.")

# ---------- load model with robust tokenizer fallback ----------
@st.cache_resource
def load_tokenizer_and_model():
    # Strategy:
    # 1) Try tokenizer from adapter folder (use_fast=True)
    # 2) If that fails, try adapter folder with use_fast=False
    # 3) If still fails, fall back to base model tokenizer (use_fast=True then use_fast=False)
    tokenizer = None
    model = None

    # Helper to attempt tokenizer load safely
    def try_tokenizer(path, use_fast_flag=True):
        try:
            tok = AutoTokenizer.from_pretrained(path, use_fast=use_fast_flag)
            return tok
        except Exception as e:
            return e

    # 1) Attempt tokenizer from adapter (fast)
    if os.path.isdir(LORA_PATH):
        st.info("Attempting to load tokenizer from adapter folder (use_fast=True)...")
        t = try_tokenizer(LORA_PATH, use_fast_flag=True)
        if isinstance(t, Exception):
            st.warning(f"Adapter fast tokenizer failed: {t}")
        else:
            tokenizer = t

    # 2) Try adapter tokenizer with use_fast=False (SentencePiece / fallback)
    if tokenizer is None and os.path.isdir(LORA_PATH):
        st.info("Attempting to load tokenizer from adapter folder (use_fast=False)...")
        t = try_tokenizer(LORA_PATH, use_fast_flag=False)
        if isinstance(t, Exception):
            st.warning(f"Adapter slow tokenizer failed: {t}")
        else:
            tokenizer = t

    # 3) Fallback to base tokenizer (fast then slow)
    if tokenizer is None:
        st.info(f"Falling back to base tokenizer: {BASE_MODEL}")
        t = try_tokenizer(BASE_MODEL, use_fast_flag=True)
        if isinstance(t, Exception):
            st.warning(f"Base fast tokenizer failed: {t}\nTrying base tokenizer with use_fast=False...")
            t = try_tokenizer(BASE_MODEL, use_fast_flag=False)
            if isinstance(t, Exception):
                # give up
                st.error("Failed to load any tokenizer (adapter or base). Please ensure tokenizer files (tokenizer.json or spiece.model) are present in the adapter folder or the base model is available.")
                raise RuntimeError("Tokenizer load failed: " + str(t))
            else:
                tokenizer = t
        else:
            tokenizer = t

    # 4) Load base model (float) and attach LoRA if possible
    st.info(f"Loading base model: {BASE_MODEL} (this may take a while)...")
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    base = base.to(DEVICE)

    # Attach LoRA adapter if available and compatible
    if os.path.isdir(LORA_PATH):
        try:
            st.info("Attaching LoRA adapter from local folder...")
            model_with_adapter = PeftModel.from_pretrained(base, LORA_PATH)
            model_with_adapter = model_with_adapter.to(DEVICE)
            model = model_with_adapter
        except Exception as e:
            st.warning(f"Could not attach LoRA adapter (possible base-model mismatch). Error: {e}")
            st.info("Falling back to base model without adapter.")
            model = base
    else:
        model = base

    return tokenizer, model

# Attempt to load and show friendly errors
try:
    tokenizer, model = load_tokenizer_and_model()
    st.success("Tokenizer and model loaded (or base model fallback).")
except Exception as e:
    st.error("Model/tokenizer loading failed. See warnings above for details.")
    # re-raise to make logs available in Streamlit logs
    raise

# ---------- simple UI to generate example questions ----------
st.header("Generate example questions")
topic = st.text_input("Enter chapter/topic (e.g., Nutrition in Plants):", value="Nutrition in Plants")
num_q = st.slider("How many questions?", 1, 10, 5)

if st.button("Generate"):
    prompt = f"You are an NCERT Class teacher. CHAPTER: {topic}\nGenerate {num_q} long-answer questions. Number them 1., 2., 3."
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
    # ensure inputs are on same device as model
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    gen = model.generate(**inputs, max_new_tokens=300, num_beams=4, repetition_penalty=1.6, no_repeat_ngram_size=3, early_stopping=True)
    out = tokenizer.decode(gen[0], skip_special_tokens=True)
    st.subheader("Raw Output")
    st.code(out[:4000])
    st.subheader("Parsed Questions")
    import re
    matches = re.findall(r"(?:^|\n)\s*(\d{1,2})[.)\-]?\s*(.+?)(?=(?:\n\s*\d{1,2}[.)\-])|\Z)", out, flags=re.S)
    if matches:
        for i, (_, q) in enumerate(matches, 1):
            q = q.strip()
            if not q.endswith("?"):
                q = q.rstrip(". ") + "?"
            st.write(f"**{i}.** {q}")
    else:
        st.write(out)
