💬 Mental Health Emotion Detector

A lightweight, interactive mental health emotion detection chat app powered by machine learning and natural language processing (NLP). Built using Python, Streamlit, and a Logistic Regression model trained on the GoEmotions dataset.

------------------------------------------------------------

🔧 Tech Stack

- Python
- NLP: NLTK, Regex, TF-IDF
- ML Model: Logistic Regression
- UI: Streamlit (Chat-style Interface)
- Data Processing: Pandas
- Visualization: Matplotlib, Seaborn

------------------------------------------------------------

🎯 Features

- Detects emotional tone from user messages
- Handles negations (e.g., "not happy" ≠ "happy")
- Flags crisis/suicidal phrases with helpful links
- Differentiates between sadness, joy, fear, and neutral tones
- Visual performance metrics (confusion matrix)
- Gradient background + emotion-colored output
- Ethical warnings and support prompts

------------------------------------------------------------

📦 Installation

python -m venv .venv
source .venv/bin/activate  (or .venv\Scripts\activate on Windows)
pip install -r requirements.txt

Also ensure nltk resources are downloaded:

import nltk
nltk.download('punkt')
nltk.download('stopwords')

------------------------------------------------------------

🚀 Usage

1. Preprocess & Train

python src/preprocessing.py     # Clean & map emotions
python src/train_model.py       # Train logistic regression model

2. Launch Chat App

streamlit run ui/app.py

Open http://localhost:8501 in your browser.

------------------------------------------------------------

🧪 Dataset Used

- GoEmotions: 58k Reddit comments labeled for 27 emotions + neutral
- Cleaned version saved to: data/processed/clean_data.csv

------------------------------------------------------------

📊 Analysis Tools

- logs/confusion_matrix.png: Model performance matrix

------------------------------------------------------------

📌 Future Additions

- BERT-based classifier
- Multi-label emotion support
- Emotion timeline during conversation
- Emotion journal download/export
- User mood graphing

------------------------------------------------------------

🤝 Contributing

Pull requests, feedback, and issue reports are welcome!

------------------------------------------------------------

📄 License

MIT License © 2025 Stuti Khanna
