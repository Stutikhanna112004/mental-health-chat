import pandas as pd
import os
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Text cleaning function (no stopword removal)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 1. Load cleaned data
df = pd.read_csv("data/processed/clean_data.csv")
df = df.dropna(subset=["clean_text"])
print(f"📄 Loaded {len(df)} samples after cleaning.")

# Optional: Show emotion distribution
print("📊 Emotion distribution:\n", df["emotion"].value_counts())

# 2. TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df["clean_text"])
y = df["emotion"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Logistic Regression (with class balancing)
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)
print("🧠 Model training complete.")

# 5. Evaluation
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# 6. Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)
os.makedirs("logs", exist_ok=True)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, xticklabels=model.classes_, yticklabels=model.classes_,
            annot=False, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("logs/confusion_matrix.png")
plt.close()
print("📈 Confusion matrix saved to logs/confusion_matrix.png")

# 7. Save model and vectorizer
os.makedirs("models", exist_ok=True)
with open("models/logreg_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("✅ Model and vectorizer saved to models/")
