import streamlit as st
from transformers import pipeline
import torch

# -------------------------------
# CACHE MODEL (Loads only once)
# -------------------------------
@st.cache_resource
def load_model():
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    return sentiment_pipeline

model = load_model()

# -------------------------------
# UI
# -------------------------------
st.title("📺 YouTube Comment Sentiment Analyzer (Transformers)")
st.markdown("This app uses **BERT (DistilBERT)** for sentiment analysis.")

user_text = st.text_area(
    "Enter a YouTube comment to analyze:",
    placeholder="I loved this video!"
)

if st.button("Predict Sentiment"):
    if user_text.strip() != "":
        result = model(user_text)[0]

        label = result["label"]
        score = result["score"]

        st.subheader(f"Sentiment: **{label}**")
        st.write(f"Confidence Score: **{score:.4f}**")

        if label == "POSITIVE":
            st.success("😊 Positive Comment")
        else:
            st.error("😡 Negative Comment")

    else:
        st.warning("Please enter some text first.")
