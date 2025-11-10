import os
import random
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from emotion_model import get_emotion

# --- Setup ---
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")  # optional (Hugging Face API key if available)

app = Flask(__name__)

# In-memory user states
user_states = {}

STRESS_QUESTIONS = [
    "Do you often feel overwhelmed or tense? (Often / Sometimes / Rarely)",
    "Do you have trouble relaxing or sleeping? (Often / Sometimes / Rarely)",
    "Do you find it hard to focus on tasks? (Often / Sometimes / Rarely)"
]

# Fallback emotion replies
FALLBACK_RESPONSES = {
    "joy": ["That’s wonderful to hear! 😊", "I’m glad something made you happy today!"],
    "love": ["Love brings warmth. 💚", "That’s lovely—cherish that connection!"],
    "sadness": ["It’s okay to not be okay 💙. What’s been on your mind?", "I hear you. What’s been hardest lately?"],
    "fear": ["It sounds unsettling. Let’s take a deep breath together.", "You’re safe now. Want to ground yourself a bit?"],
    "anger": ["It’s valid to feel angry. Want to unpack it a little?", "Let’s take a second—what’s triggering it most?"],
    "neutral": ["I’m here. What’s been on your mind today?", "Tell me more—I’m listening."]
}

# --- GPT Reply ---
def gpt_reply(user_text: str, emotion: str) -> str:
    """Try GPT; if fails, return None (so fallback triggers)."""
    if not OPENAI_KEY:
        return None

    crisis_terms = ["suicide", "kill myself", "end my life", "self harm"]
    if any(term in user_text.lower() for term in crisis_terms):
        return (
            "I'm really glad you told me this. Your safety matters deeply. 💛 "
            "If you’re in danger, please call 112 (India) or reach out to someone nearby right now."
        )

    system_prompt = (
        "You are MindMate, an empathetic AI wellness companion for youth. "
        "Be warm, understanding, and emotionally intelligent. "
        "Speak naturally in 1–3 short sentences, and suggest small coping steps like breathing or journaling. "
        "Avoid robotic or clinical tone."
    )

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Emotion: {emotion}. Message: {user_text}"}
        ],
        "temperature": 0.8,
        "max_tokens": 150
    }

    try:
        res = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}"},
            json=payload,
            timeout=20
        )
        if res.status_code != 200:
            print(f"⚠️ GPT ERROR {res.status_code}: {res.text}")
            return None
        data = res.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"⚠️ GPT ERROR: {e}")
        return None


# --- Hugging Face Fallback ---
def hf_fallback_reply(user_text: str, emotion: str) -> str:
    """Generate conversational replies using Hugging Face API (free fallback)."""
    print("⚙️ Using Hugging Face fallback reply...")

    model_url = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    try:
        res = requests.post(model_url, headers=headers, json={"inputs": user_text}, timeout=20)
        data = res.json()
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        elif isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
            return data[0]["generated_text"]
    except Exception as e:
        print(f"⚠️ HF ERROR: {e}")

    # If API fails → fallback generic response
    return random.choice(FALLBACK_RESPONSES.get(emotion, FALLBACK_RESPONSES["neutral"]))


# --- Stress Test Logic ---
def score_stress(answers):
    total = 0
    for a in answers:
        t = a.lower()
        if "often" in t: total += 3
        elif "sometimes" in t: total += 2
        elif "rarely" in t: total += 1
    return total

def stress_recommendation(score):
    if score >= 7:
        return {
            "level": "High",
            "advice": (
                "Your stress seems high 😟. Try 3–5 minutes of deep breathing, "
                "go for a walk, or talk to someone you trust."
            ),
            "actions": [
                "Try box breathing (4-4-4-4)",
                "Step outside for fresh air",
                "Write one thought you want to release"
            ]
        }
    elif score >= 4:
        return {
            "level": "Moderate",
            "advice": (
                "Some tension shows up—try stretching, slow breathing, or journaling for a few minutes 💪."
            ),
            "actions": [
                "Stretch your body gently",
                "Take 10 slow breaths",
                "Write 3 things you're grateful for"
            ]
        }
    else:
        return {
            "level": "Low",
            "advice": (
                "Nice—stress looks manageable! Keep up your balance of rest and small joys."
            ),
            "actions": [
                "Keep a gratitude journal",
                "Stay hydrated",
                "Plan one small treat for yourself today"
            ]
        }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_id = "default_user"
    text = request.form.get("message", "").strip()
    if not text:
        return jsonify({"reply": "Could you share that again?", "type": "chat"})

    # Initialize user state
    if user_id not in user_states:
        user_states[user_id] = {"stage": None, "answers": [], "offered_stress": False}

    state = user_states[user_id]

    # --- Stress Test Flow ---
    if state["stage"] == "stress":
        state["answers"].append(text)
        if len(state["answers"]) >= 3:
            score = score_stress(state["answers"])
            rec = stress_recommendation(score)
            user_states[user_id] = {"stage": None, "answers": [], "offered_stress": False}
            reply = (
                f"🧘 Stress Level: {rec['level']}\n{rec['advice']}\n"
                f"Suggested actions: • {rec['actions'][0]} • {rec['actions'][1]} • {rec['actions'][2]}"
            )
            return jsonify({"reply": reply, "type": "result"})
        else:
            return jsonify({"reply": STRESS_QUESTIONS[len(state['answers'])], "type": "stress"})

    # Emotion detection
    emotion = get_emotion(text)

    # Offer stress test
    if any(k in text.lower() for k in ["not good", "sad", "depressed", "stressed", "tired", "anxious"]):
        state["offered_stress"] = True
        return jsonify({"reply": "It sounds tough 😔. Want to take a quick 3-question stress check?", "type": "offer_test"})

    # Start stress test
    if any(k in text.lower() for k in ["yes", "sure", "ok", "start"]) and state.get("offered_stress"):
        state["stage"] = "stress"
        state["answers"] = []
        return jsonify({"reply": STRESS_QUESTIONS[0], "type": "stress"})

    # Relaxation resources
    if any(k in text.lower() for k in ["relax", "calm", "anger", "breathe", "meditate"]):
        reply = (
            "Here are some quick relaxation tools 🌿:\n"
            "• [10-Minute Guided Relaxation (YouTube)](https://www.youtube.com/watch?v=inpok4MKVLM)\n"
            "• [Gratitude Journaling Tips (Article)](https://positivepsychology.com/gratitude-journal/)\n"
            "• [Positive Thinking Audio (Spotify)](https://open.spotify.com/track/6dGnYIeXmHdcikdzNNDMm2)\n"
        )
        return jsonify({"reply": reply, "type": "resource"})

    # --- Generate Reply (GPT → HF Fallback) ---
    reply = gpt_reply(text, emotion)
    if not reply:
        reply = hf_fallback_reply(text, emotion)

    return jsonify({"reply": reply, "type": "chat"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
