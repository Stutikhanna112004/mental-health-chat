from datasets import load_dataset
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords


nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# GOemotion labels
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness',
    'surprise', 'neutral'
]

def clean_text(text):
    """
    Cleans input text by removing URLs, punctuation, stopwords, and lowercasing.
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove punctuation and numbers
    text = re.sub(r"\s+", " ", text).strip()  # remove extra whitespace
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def load_goemotions():
    """
    Loads the GoEmotions dataset using Hugging Face Datasets.
    Saves raw dataset to CSV.
    """
    print("🔽 Loading GoEmotions dataset...")
    dataset = load_dataset("go_emotions", split="train")
    df = pd.DataFrame(dataset)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/go_emotions.csv", index=False)
    print(f"✅ Raw dataset saved to data/go_emotions.csv with {len(df)} rows.")
    return df

def preprocess_dataset(df):
    """
    Applies text cleaning and emotion label mapping.
    Saves the processed dataset.
    """
    print("🧹 Preprocessing text and mapping emotion labels...")

    
    df['clean_text'] = df['text'].apply(clean_text)

    
    df['emotion'] = df['labels'].apply(lambda ids: EMOTION_LABELS[ids[0]] if ids else 'neutral')

    
    df = df[['clean_text', 'emotion']]

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/clean_data.csv", index=False)
    print("✅ Cleaned data saved to data/processed/clean_data.csv")
    return df


if __name__ == "__main__":
    df = load_goemotions()
    clean_df = preprocess_dataset(df)
    print(clean_df.head())
