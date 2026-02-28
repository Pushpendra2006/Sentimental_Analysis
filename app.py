import streamlit as st
from transformers import pipeline

# Load fine-tuned model
@st.cache_resource
def load_model():
    classifier = pipeline(
        "text-classification",
        model="pushpendra2006/sentimental_analysis",
        token=None   # only if public
    )
    return classifier
model = load_model()

st.title("📺 YouTube Comment Sentiment Analyzer (BERT - 3 Class)")
st.markdown("Predicts **Positive / Neutral / Negative** sentiment.")

user_text = st.text_area("Enter a comment:")

if st.button("Predict"):
    if user_text.strip() != "":
        result = model(user_text)[0]
        
        label = result["label"]
        score = result["score"]

        st.subheader(f"Sentiment: **{label}**")
        st.write(f"Confidence: **{score:.4f}**")

        if label == "LABEL_0":
            st.error("😡 Negative")
        elif label == "LABEL_1":
            st.info("😐 Neutral")
        else:
            st.success("😊 Positive")

    else:
        st.warning("Please enter text.")

