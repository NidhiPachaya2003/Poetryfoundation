import streamlit as st
import numpy as np
import re
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("text_model.keras")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# UI
st.title("🔥 Poetry Author Predictor")

text = st.text_area("Enter your text here:")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter text!")
    else:
        cleaned = clean_text(text)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=100)

        pred = model.predict(padded)
        label = le.inverse_transform([np.argmax(pred)])[0]

        st.success(f"Author: {label}")