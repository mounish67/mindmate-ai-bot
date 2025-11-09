import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Hugging Face API key (from environment or direct insert)
HF_TOKEN = os.getenv("HF_API_KEY")

API_URL = "https://api-inference.huggingface.co/models/bhadresh-savani/distilbert-base-uncased-emotion"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def get_emotion(text: str) -> str:
    """
    Send text to Hugging Face API and return the dominant emotion label.
    """
    if not HF_TOKEN:
        # fallback if no key
        return "neutral"

    payload = {"inputs": text}
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=10)
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            emotions = data[0]
            best = max(emotions, key=lambda x: x["score"])
            return best["label"].lower()
        else:
            return "neutral"
    except Exception:
        return "neutral"
