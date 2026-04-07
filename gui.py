import tkinter as tk
from tkinter import messagebox
import numpy as np
import re
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model & files
model = load_model("text_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# Prediction function
def predict_text():
    text = entry.get("1.0", tk.END)

    if text.strip() == "":
        messagebox.showwarning("Warning", "Please enter text!")
        return

    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=100)

    pred = model.predict(padded)
    label = le.inverse_transform([np.argmax(pred)])[0]

    result_label.config(text=f"Prediction: {label}")

# GUI window
root = tk.Tk()
root.title("✨ AI Text Classifier")
root.geometry("600x450")
root.config(bg="#1e1e2f")

# Title
title = tk.Label(
    root,
    text="AI Text Classification App",
    font=("Helvetica", 18, "bold"),
    bg="#1e1e2f",
    fg="#00ffcc"
)
title.pack(pady=15)

# Text box
entry = tk.Text(
    root,
    height=8,
    width=60,
    font=("Arial", 12),
    bg="#2c2c3e",
    fg="white",
    insertbackground="white"
)
entry.pack(pady=10)

# Predict button
btn = tk.Button(
    root,
    text="🚀 Predict",
    command=predict_text,
    font=("Arial", 12, "bold"),
    bg="#00cc99",
    fg="black",
    padx=10,
    pady=5
)
btn.pack(pady=15)

# Result label
result_label = tk.Label(
    root,
    text="Prediction will appear here",
    font=("Arial", 14, "bold"),
    bg="#1e1e2f",
    fg="#ffffff"
)
result_label.pack(pady=20)

# Footer
footer = tk.Label(
    root,
    text="Developed using Keras + NLP",
    font=("Arial", 9),
    bg="#1e1e2f",
    fg="gray"
)
footer.pack(side="bottom", pady=10)

# Run app
root.mainloop()