import pandas as pd
import numpy as np
import re
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
df = pd.read_csv("kaggle_poem_dataset.csv")

# Select required columns
df = df[['Content', 'Author']].dropna()

# Remove Anonymous authors
df = df[df['Author'] != 'Anonymous']

# Remove authors with very few samples (improves accuracy)
df = df.groupby('Author').filter(lambda x: len(x) > 50)

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

df['Content'] = df['Content'].apply(clean_text)

texts = df['Content'].astype(str)
labels = df['Author'].astype(str)

# Tokenization
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=100)

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    padded, labels_encoded, test_size=0.2, random_state=42
)

# Model (LSTM - better accuracy)
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=100),
    LSTM(128, return_sequences=False),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Save files
model.save("text_model.keras")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ ALL FILES SAVED SUCCESSFULLY")