import streamlit as st
import pickle
import re
import numpy as np
from nltk.corpus import stopwords
from gensim.models import Word2Vec

# Load models
model = pickle.load(open("svm_model.pkl", "rb"))
w2v_model = Word2Vec.load("w2v_vectorizer.pkl")   # ✅ correct file

stop_words = set(stopwords.words('english'))

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return words   # ⚠️ return tokens (important)

# Convert sentence → vector
def sentence_vector(tokens, model):
    vectors = []
    for word in tokens:
        if word in model.wv:
            vectors.append(model.wv[word])
    
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    
    return np.mean(vectors, axis=0)

# Prediction function
def predict_spam(text):
    tokens = clean_text(text)
    vec = sentence_vector(tokens, w2v_model).reshape(1, -1)
    pred = model.predict(vec)
    return "🚨 Spam" if pred[0] == 1 else "✅ Not Spam"

# UI
st.title("📧 Spam Detection App (Word2Vec + SVM)")

user_input = st.text_area("Enter your message:")

if st.button("Predict"):
    if user_input.strip() != "":
        result = predict_spam(user_input)
        st.success(result)
    else:
        st.warning("Please enter a message")