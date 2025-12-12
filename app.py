import streamlit as st
import pandas as pd
import random

st.set_page_config(page_title="NCERT Teacher-Style Question Generator", layout="wide")

# -------------------------
# Load CSV
# -------------------------
CSV_PATH = "ncert_teacher_questions.csv"   # rename your CSV to this or change name here

st.title("📘 NCERT Teacher-Style Question Generator")
st.write("Generate high-quality long-answer teacher-style questions using your dataset.")

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    df["Class"] = df["Class"].astype(str)
    return df

try:
    df = load_data()
except:
    st.error("CSV file not found. Upload a CSV file.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.stop()

# Sidebar Filters
with st.sidebar:
    st.header("Filters")
    class_list = sorted(df["Class"].unique().tolist())
    selected_class = st.selectbox("Select Class", class_list)

    chapter_list = sorted(df[df["Class"] == selected_class]["Chapter"].unique().tolist())
    selected_chapter = st.selectbox("Select Chapter", chapter_list)

    num_q = st.slider("How many questions to generate?", 1, 10, 5)

# Filter dataset
subset = df[(df["Class"] == selected_class) & (df["Chapter"] == selected_chapter)]

# Template variations
TEMPLATES = [
    "Explain in detail: {}",
    "Describe the key ideas related to '{}'.",
    "Discuss important concepts from '{}'. Provide examples.",
    "Write a detailed note on '{}'.",
    "What are the major learning points of '{}'? Explain.",
    "How does '{}' relate to real-life applications?",
    "Summarize the concept of '{}', highlighting its importance.",
]

def generate_variation(base_q):
    """Create a question variation using templates."""
    template = random.choice(TEMPLATES)
    topic = base_q.split("?")[0].strip()
    return template.format(topic) + "?"

# -------------------------
# Generate Questions
# -------------------------
st.subheader(f"Questions for Class {selected_class} — {selected_chapter}")

if st.button("Generate Questions"):
    if len(subset) == 0:
        st.error("No questions found for this chapter in your CSV.")
        st.stop()

    base_questions = subset["Question"].sample(min(num_q, len(subset))).tolist()

    final_questions = []

    for q in base_questions:
        if len(final_questions) >= num_q:
            break
        new_q = generate_variation(q)
        final_questions.append(new_q)

    st.success("Generated Questions:")
    for i, q in enumerate(final_questions, start=1):
        st.write(f"**{i}. {q}**")

    # Download
    out_df = pd.DataFrame({
        "Class": [selected_class] * len(final_questions),
        "Chapter": [selected_chapter] * len(final_questions),
        "Question": final_questions
    })

    st.download_button(
        label="Download as CSV",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name="generated_questions.csv"
    )

else:
    st.info("Click **Generate Questions** to create teacher-style questions.")
