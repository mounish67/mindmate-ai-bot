import os
import random
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from emotion_model import get_emotion

# --- Setup ---
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# In-memory state
# { "default_user": {"stage": None|"stress", "answers": [], "context": [], "offered_stress": False} }
user_states = {}

STRESS_QUESTIONS = [
    "Do you often feel overwhelmed or tense? (Often / Sometimes / Rarely)",
    "Do you have trouble relaxing or sleeping? (Often / Sometimes / Rarely)",
    "Do you find it hard to focus on tasks? (Often / Sometimes / Rarely)"
]

# Fallbacks if GPT fails
FALLBACK_RESPONSES = {
    "joy": [
        "That’s lovely to hear! What made your day brighter?",
        "I love that energy—want to share what went well? 😊"
    ],
    "love": [
        "Love brings warmth—hold onto that feeling. 💚",
        "That sounds meaningful. Who or what inspired it?"
    ],
    "sadness": [
        "I’m sorry it’s heavy right now. Want to talk about what’s weighing on you?",
        "It’s okay to not be okay—what would feel supportive right now?"
    ],
    "fear": [
        "That sounds unsettling. Let’s slow down—what’s the biggest worry?",
        "You’re not alone. Would grounding or breathing help?"
    ],
    "anger": [
        "It’s valid to feel angry. What triggered it?",
        "Let’s unpack it—what might help release that tension?"
    ],
    "surprise": [
        "Whoa, that’s unexpected! How are you feeling about it now?",
        "Surprises can throw us off—good or tough kind?"
    ],
    "neutral": [
        "I’m here—what’s been on your mind today?",
        "Tell me more; I’m listening."
    ]
}

# --- GPT Reply (HTTP API, robust) ---
def gpt_reply(user_text: str, emotion: str) -> str:
    """Use OpenAI Chat Completions via HTTP; handle all common errors gracefully."""
    fallback = random.choice(FALLBACK_RESPONSES.get(emotion, FALLBACK_RESPONSES["neutral"]))

    if not OPENAI_KEY:
        print("⚠️ GPT ERROR: OPENAI_API_KEY not set")
        return fallback

    # Crisis safety
    crisis_terms = ["suicide", "kill myself", "end my life", "self harm", "cut myself", "hurt myself"]
    if any(term in user_text.lower() for term in crisis_terms):
        return (
            "I'm really glad you told me this. Your safety matters deeply. 💛 "
            "If you’re in danger, please call your local emergency number (e.g., 112 in India) "
            "or reach out to a trusted person nearby right now."
        )

    system_prompt = (
        "You are MindMate, an empathetic AI wellness companion for youth. "
        "Respond naturally in 1–3 sentences. Be emotionally aware, supportive, and realistic. "
        "Encourage small healthy actions (breathing, walking, journaling, grounding). "
        "Avoid clinical/diagnostic language. Keep tone warm and human, not robotic."
    )

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Detected emotion: {emotion}. User said: {user_text}"}
        ],
        "temperature": 0.8,
        "max_tokens": 150
    }

    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=20
        )

        if r.status_code != 200:
            print(f"⚠️ GPT HTTP {r.status_code}: {r.text}")
            return fallback

        data = r.json()

        # Normal shape
        if isinstance(data, dict) and "choices" in data and data["choices"]:
            content = data["choices"][0].get("message", {}).get("content", "")
            if content:
                return content.strip()

        # Unexpected shape (SDK/endpoint variations)
        print(f"⚠️ GPT unexpected payload: {data}")
        return fallback

    except Exception as e:
        print(f"⚠️ GPT ERROR: {e}")
        return fallback

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
                "Your stress seems high 😟. Try 3–5 minutes of deep breathing (inhale 4s, hold 4s, exhale 6s), "
                "go for a walk, or write down what feels heavy. Talk to someone you trust."
            ),
            "actions": [
                "Do box-breathing (4-4-4-4)",
                "Step outside for 5 minutes",
                "Call or text a trusted friend"
            ]
        }
    elif score >= 4:
        return {
            "level": "Moderate",
            "advice": (
                "Some tension shows up—try 10 slow breaths or gentle stretching. "
                "Finish a small task to regain focus 💪."
            ),
            "actions": [
                "Breathe deeply 10 times",
                "Stretch your arms and back",
                "Write 3 things you're grateful for"
            ]
        }
    else:
        return {
            "level": "Low",
            "advice": (
                "Nice—stress looks manageable! Keep up your habits like good sleep, hydration, and small joys."
            ),
            "actions": [
                "Do a 5-min gratitude note",
                "Drink water",
                "Plan one small enjoyable activity today"
            ]
        }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_id = "default_user"  # simple single-user demo; can be replaced with sessions later
    text = request.form.get("message", "").strip()
    if not text:
        return jsonify({"reply": "Could you share that again?", "type": "chat"})

    # Init state
    if user_id not in user_states:
        user_states[user_id] = {"stage": None, "answers": [], "context": [], "offered_stress": False}

    state = user_states[user_id]
    state["context"].append(text)
    if len(state["context"]) > 8:
        state["context"] = state["context"][-8:]

    # Reset/clear command
    if any(k in text.lower() for k in ["restart", "reset", "clear chat", "start over"]):
        user_states[user_id] = {"stage": None, "answers": [], "context": [], "offered_stress": False}
        return jsonify({"reply": "Chat cleared 🌱. Hey there! How are you feeling today?", "type": "chat"})

    # Ongoing stress test flow
    if state["stage"] == "stress":
        state["answers"].append(text)
        if len(state["answers"]) >= 3:
            score = score_stress(state["answers"])
            rec = stress_recommendation(score)
            user_states[user_id] = {"stage": None, "answers": [], "context": [], "offered_stress": False}
            reply = (
                f"🧘 Stress Level: {rec['level']}\n"
                f"{rec['advice']}\n"
                f"Suggested actions: • {rec['actions'][0]} • {rec['actions'][1]} • {rec['actions'][2]}"
            )
            return jsonify({"reply": reply, "type": "result"})
        else:
            return jsonify({"reply": STRESS_QUESTIONS[len(state['answers'])], "type": "stress"})

    # Emotion detection
    emotion = get_emotion(text)

    # Offer stress test if negative cues
    if any(k in text.lower() for k in ["not good", "sad", "depressed", "anxious", "angry",
                                       "stressed", "moody", "tired", "overwhelmed"]):
        state["offered_stress"] = True
        return jsonify({"reply": "It sounds tough 😔. Want to take a quick 3-question stress check?", "type": "offer_test"})

    # Start test if user agrees and it was offered recently
    if any(k in text.lower() for k in ["yes", "sure", "ok", "start", "take test", "let's do it", "yeah"]):
        if state.get("offered_stress", False):
            state["stage"] = "stress"
            state["answers"] = []
            state["offered_stress"] = False
            return jsonify({"reply": STRESS_QUESTIONS[0], "type": "stress"})

    # Smart relaxation suggestions
    if any(k in text.lower() for k in ["relax", "stress relief", "calm", "meditate", "breathe", "anger control"]):
        reply = (
            "Here are some quick relaxation tools 🌿:\n\n"
            "• [10-Minute Guided Relaxation (YouTube)](https://www.youtube.com/watch?v=inpok4MKVLM)\n"
            "• [Gratitude Journaling Tips (Article)](https://positivepsychology.com/gratitude-journal/)\n"
            "• [Positive Thinking Audio (Spotify)](https://open.spotify.com/track/6dGnYIeXmHdcikdzNNDMm2)\n\n"
            "Would you like to take the short stress test too?"
        )
        return jsonify({"reply": reply, "type": "resource"})

    # Otherwise, GPT reply (real-time)
    reply = gpt_reply(text, emotion)
    return jsonify({"reply": reply, "type": "chat"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)  # keep debug off in production
