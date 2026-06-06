import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    classifier = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        return_all_scores=False
    )
    return classifier

model = load_model()
st.title("YouTube Comment Sentiment Analyzer")
st.markdown("Predicts **Positive / Neutral / Negative** sentiment using Transformers.")

user_text = st.text_area(
    "Enter a YouTube comment:",
    placeholder="This video was okay."
)

if st.button("Predict Sentiment"):
    if user_text.strip():
        result = model(user_text)[0]
        label = result["label"]
        score = result["score"]

        # Model label mapping
        if label == "LABEL_0":
            sentiment = "Negative 😡"
            st.error(f"Sentiment: {sentiment}")
        elif label == "LABEL_1":
            sentiment = "Neutral 😐"
            st.info(f"Sentiment: {sentiment}")
        else:
            sentiment = "Positive 😊"
            st.success(f"Sentiment: {sentiment}")

        st.write(f"Confidence Score: **{score:.4f}**")

    else:
        st.warning("Please enter some text first.")
