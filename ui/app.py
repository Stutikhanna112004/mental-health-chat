import streamlit as st
import pickle
import os
import re


st.markdown("""
<style>
body {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    background-attachment: fixed;
    background-size: cover;
    color: white;
}
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
    color: white;
}

h1, h2, h3 {
    color: #ffffff;
}

input, .stTextInput>div>div>input {
    background-color: #ffffff20;
    color: white;
    border-radius: 10px;
    border: 1px solid #ffffff33;
    padding: 0.75rem;
}

.result-box {
    background-color: #ffffff22;
    padding: 1rem;
    border-radius: 1rem;
    border-left: 6px solid #facc15;
    margin-top: 1rem;
    color: #fff;
}
</style>
""", unsafe_allow_html=True)

# --- Text cleaning ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "models", "logreg_model.pkl")
    vectorizer_path = os.path.join(base_dir, "..", "models", "tfidf_vectorizer.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()


st.set_page_config(page_title="Mental Health Chat", layout="centered")
st.title("💬 Mental Health Emotion Detector")
st.markdown("Welcome. Type anything you want — we'll try to understand how you're feeling. 💙")


user_input = st.text_input("You:", placeholder="Type your message here...", key="user")

if user_input:
    user_text = clean_text(user_input)


    crisis_keywords = [
        "suicidal", "kill myself", "want to die", "end my life", "i want to die", "i hate my life", "i give up", "can't go on", "i feel like ending it all", "i want to die","i dont want to live anymore", "i wish i never woke up", "i'm thinking about ending it", "no one would care if i disappeared", "it would be better if i was gone", "i cant do this anymore", "i want to disappear forever", "i'm done with life", "i'm tired of being alive","i want to kill myself","i feel like giving up","everything is pointless","i just want the pain to stop","im not safe right now", "life is meaningless", "what's the point of anything","i feel like ending it all", "i hate my life", "i shouldn't be here"
    ]
    if any(kw in user_text for kw in crisis_keywords):
        st.markdown(f"""
        <div class='result-box'>
            🚨 <strong>Crisis Alert</strong>: You're not alone. Please talk to someone you trust or a professional.<br><br>
            📞 <a href='https://icallhelpline.org/' target='_blank'>Click here to find help</a>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    
    neutral_phrases = ["im okay", "just another day", "nothing much", "not sad", "not angry", "it is what it is",
    "nothing much",
    "i dont feel anything",
    "its whatever",
    "i guess im fine",
    "not feeling anything in particular",
    "im used to it",
    "it is what it is",
    "nothing to say really",
    "i'm not sure how i feel",
    "just going through the motions",
    "can't complain",
    "i'm functioning"]
    sadness_phrases = ["im not happy", "i felt like crying", "i feel invisible", "i'm so tired of everything",  "i feel like crying",
    "no one understands me",
    "i'm so tired of everything",
    "everything feels heavy",
    "i feel empty inside",
    "i miss how things used to be",
    "im not good enough",
    "i feel lost",
    "why does it always happen to me",
    "i just want to disappear",
    "i dont belong anywhere",
    "i hate how i feel",
    "my chest feels heavy",
    "i feel invisible",
    "nothing makes me happy anymore"]

    if any(p in user_text for p in sadness_phrases):
        prediction = "sadness"
        confidence = 99.0
    elif any(p in user_text for p in neutral_phrases):
        prediction = "neutral"
        confidence = 98.0
    else:
        X_input = vectorizer.transform([user_text])
        prediction = model.predict(X_input)[0]
        confidence = round(model.predict_proba(X_input).max() * 100, 2)

    
    emotion_icons = {
        "joy": "😄", "sadness": "😢", "anger": "😡", "fear": "😨",
        "gratitude": "🙏", "love": "❤️", "grief": "💔", "surprise": "😲",
        "neutral": "😐", "approval": "👍", "disapproval": "👎",
        "amusement": "😆", "remorse": "😞", "confusion": "🤔", "disappointment": "😞"
    }
    icon = emotion_icons.get(prediction, "💬")

    # --- Output ---
    # Define emotion colors
    emotion_colors = {
    "joy": "#22c55e",          # green
    "sadness": "#3b82f6",      # blue
    "anger": "#ef4444",        # red
    "fear": "#8b5cf6",         # purple
    "gratitude": "#10b981",    # teal
    "love": "#ec4899",         # pink
    "grief": "#64748b",        # slate
    "surprise": "#f59e0b",     # amber
    "neutral": "#9ca3af",      # gray
    "approval": "#16a34a",     # green-dark
    "disapproval": "#dc2626",  # red-dark
    "amusement": "#eab308",    # yellow
    "remorse": "#0ea5e9",      # cyan
    "confusion": "#f472b6",    # rose
    "disappointment": "#6b7280" # cool gray
}


    color = emotion_colors.get(prediction, "#8899af")


    st.markdown(f"""
    <div class="result-box">
                 <strong>Detected Emotion:</strong> <span style="color:{color}; font-weight:700;">{icon} {prediction}</span><br>
                 🔍 <span style="font-size:0.9em;">{confidence}% confidence</span>
</div>
""", unsafe_allow_html=True)

    
    if prediction in ["sadness", "anger", "fear", "grief", "remorse", "disappointment"]:
        st.info("🧘 Take a deep breath. You're not alone — reach out to someone if you're struggling 💙")
    elif prediction in ["joy", "gratitude", "love", "amusement"]:
        st.success("😊 That's beautiful. Keep the good vibes going!")

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 10px;
        left: 0;
        width: 100%;
        text-align: center;
        font-size: 0.85rem;
        color: #ccc;
        font-family: 'Segoe UI', sans-serif;
    }
    .footer a {
        color: #facc15;
        text-decoration: none;
    }
    </style>

    <div class="footer">
        ⚠️ This is an educational tool. Not a substitute for medical care. <br>
         Licensed under the <a href="https://opensource.org/licenses/MIT" target="_blank">MIT License</a> &copy; 2025 STUTI KHANNA
    </div>
    """,
    unsafe_allow_html=True
)
