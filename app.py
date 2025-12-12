import streamlit as st
import pandas as pd

st.set_page_config(page_title="NCERT Teacher Question Generator", layout="wide")

st.title("📘 NCERT Teacher-Style Question Generator (CSV Version)")
st.write("Upload your CSV file to explore questions chapter-wise. CSV must contain columns: Class, Chapter, Question.")

# -------------------------
# Upload CSV
# -------------------------
uploaded = st.file_uploader("Upload CSV (must contain Class, Chapter, Question columns)", type=["csv"])

if uploaded is None:
    st.info("➡ Please upload the CSV file to proceed. You can upload 'ncert_teacher_questions.csv' or any CSV with columns: Class, Chapter, Question.")
    st.stop()

# Load CSV
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error("Error reading CSV: " + str(e))
    st.stop()

# Validate columns
required_cols = {"Class", "Chapter", "Question"}
if not required_cols.issubset(df.columns):
    st.error(f"CSV must contain columns: {required_cols}")
    st.stop()

# -------------------------
# UI Filters
# -------------------------
classes = sorted(df["Class"].astype(str).unique())
selected_class = st.selectbox("Select Class", classes)

chapters = sorted(df[df["Class"].astype(str) == str(selected_class)]["Chapter"].unique())
selected_chapter = st.selectbox("Select Chapter", chapters)

filtered = df[(df["Class"].astype(str) == str(selected_class)) & (df["Chapter"] == selected_chapter)]

st.subheader(f"📖 Questions for Chapter: {selected_chapter}")

for idx, row in filtered.reset_index(drop=True).iterrows():
    st.markdown(f"**{idx+1}. {row['Question']}**")
