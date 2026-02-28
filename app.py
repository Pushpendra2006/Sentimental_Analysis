import streamlit as st
import pandas as pd
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- STEP 1: CACHED TRAINING LOGIC ---
@st.cache_resource # This ensures the model only trains ONCE when the app starts
def initialize_app():
    # Loading dataset from your folder
    df = pd.read_csv('YoutubeCommentsDataSet.csv') 
    df = df.dropna().drop_duplicates() 

    # Encoding Labels
    le = LabelEncoder() 
    df['Sentiment'] = le.fit_transform(df['Sentiment'])

    # Preprocessing Setup
    stop_words = set(stopwords.words('english'))

    def preprocessing(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))  
        text = ' '.join(word for word in text.split() if word not in stop_words)  
        return text

    df['Comment'] = df['Comment'].apply(preprocessing)

    # Training the Model (Your Logistic Regression + TF-IDF)
    x = df['Comment']
    y = df['Sentiment']
    
    # We use a fixed random state so the results are consistent
    tfidf = TfidfVectorizer(max_features=120)
    x_tfidf = tfidf.fit_transform(x)
    
    lr = LogisticRegression()
    lr.fit(x_tfidf, y)
    
    return lr, tfidf, le, preprocessing

# Initialize the model and tools
model, vectorizer, encoder, clean_func = initialize_app()

# --- STEP 2: STREAMLIT USER INTERFACE ---
st.title("📺 YouTube Comment Sentiment Analyzer")
st.markdown("This app uses **Logistic Regression** to predict comment sentiment.")

# User Input Section
user_text = st.text_area("Enter a YouTube comment to analyze:", placeholder="I loved this video!")

if st.button("Predict Sentiment"):
    if user_text.strip() != "":
        # Process the input using the same logic as training
        cleaned_input = clean_func(user_text)
        vectorized_input = vectorizer.transform([cleaned_input])
        
        # Make Prediction
        prediction = model.predict(vectorized_input)
        
        # Convert number back to text (Positive/Negative/Neutral)
        result_text = encoder.inverse_transform(prediction)[0]
        
        # Display Result
        st.subheader(f"The sentiment is: **{result_text}**")
    else:
        st.warning("Please enter some text first.")