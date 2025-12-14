import streamlit as st
import pandas as pd
import random

st.set_page_config(page_title="NCERT Question Generator", layout="wide")
st.title("📘 NCERT Chapter-wise Question Generator")

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    # Normalize columns
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df

uploaded_file = st.file_uploader("📂 Upload Questions CSV", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)

    st.subheader("🧾 Detected CSV Columns")
    st.code(list(df.columns))

    # -------- REQUIRED FOR SUBJECTIVE QUESTIONS --------
    required_columns = {"chapter", "question"}

    missing = required_columns - set(df.columns)
    if missing:
        st.error(f"❌ Missing required columns: {missing}")
        st.stop()

    chapter_name = st.text_input("📖 Enter Chapter Name")
    num_questions = st.number_input("🔢 Number of Questions", 1, 50, 5)

    if st.button("🚀 Generate Questions"):
        filtered = df[df["chapter"].str.lower() == chapter_name.lower()]

        if filtered.empty:
            st.warning("⚠ No questions found for this chapter")
        else:
            sample = filtered.sample(
                min(num_questions, len(filtered)),
                random_state=random.randint(1, 9999)
            )

            st.success(f"✅ Showing {len(sample)} questions")

            for i, row in enumerate(sample.itertuples(), 1):
                st.markdown(f"### Q{i}. {row.question}")
