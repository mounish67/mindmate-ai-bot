import os
import random
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from emotion_model import get_emotion

# ── Setup ───────────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # <–– Set this in .env or Render Environment

app = Flask(__name__)

state = {"stage": None, "answers": [], "offered_stress": False, "context": []}

STRESS_QUESTIONS = [
    "Do you often feel overwhelmed or tense? (Often / Sometimes / Rarely)",
    "Do you have trouble relaxing or sleeping? (Often / Sometimes / Rarely)",
    "Do you find it hard to focus on tasks? (Often / Sometimes / Rarely)"
]

RELAXATION_SNIPPET = (
    "Here are some relaxation tools 🌿:\n"
    "• 10-Minute Guided Relaxation (YouTube): https://www.youtube.com/watch?v=inpok4MKVLM\n"
    "• Gratitude Journaling Tips (Article): https://positivepsychology.com/gratitude-journal/\n"
    "• Positive Thinking Audio (Spotify): https://open.spotify.com/track/6dGnYIeXmHdcikdzNNDMm2\n"
)

FALLBACK_RESPONSES = {
    "joy": ["That’s wonderful 😊 What made your day brighter?"],
    "love": ["That sounds meaningful 💚 What sparked that feeling?"],
    "sadness": ["I’m sorry it feels heavy. What’s weighing on you most?"],
    "fear": ["That sounds unsettling. Let’s take one slow breath together."],
    "anger": ["It’s valid to feel angry. What triggered it?"],
    "neutral": ["I’m here — what’s been on your mind today?"]
}

# ── Helpers ───────────────────────────────────────────────────────
def add_context(role: str, text: str, keep_last: int = 8):
    state["context"].append(f"{role}: {text}")
    if len(state["context"]) > keep_last:
        state["context"] = state["context"][-keep_last:]

def recent_context_text() -> str:
    return "\n".join(state["context"][-8:])

# ── GEMINI Reply ───────────────────────────────────────────────────────
def gemini_reply(user_text: str, emotion: str) -> str:
    """Send chat to Google Gemini and return a natural, empathetic reply."""
    if not GEMINI_API_KEY:
        return random.choice(FALLBACK_RESPONSES.get(emotion, FALLBACK_RESPONSES["neutral"]))

    system_prompt = (
        "You are MindMate, a warm, human-like AI mental wellness companion. "
        "Be empathetic, supportive, and natural. Use 1–3 short sentences. "
        "Validate emotions, offer a small helpful action (like deep breathing, journaling, or walking), "
        "and end with a gentle follow-up question. Avoid clinical or diagnostic terms."
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": f"{system_prompt}\n\nRecent conversation:\n{recent_context_text()}\n\nUser ({emotion}): {user_text}"
                    }
                ],
            }
        ]
    }

    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=15
        )
        data = response.json()
        return (
            data["candidates"][0]["content"]["parts"][0]["text"].strip()
            if "candidates" in data
            else random.choice(FALLBACK_RESPONSES.get(emotion, FALLBACK_RESPONSES["neutral"]))
        )
    except Exception as e:
        print(f"⚠️ GEMINI ERROR: {e}")
        return random.choice(FALLBACK_RESPONSES.get(emotion, FALLBACK_RESPONSES["neutral"]))

# ── Stress Logic ───────────────────────────────────────────────────────
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
            "advice": "Your stress seems high 😟. Try deep breathing (4-4-6), take a 5-minute walk, or talk to a friend.",
            "actions": ["Box breathing (4-4-6)", "Short walk", "Call a trusted person"]
        }
    elif score >= 4:
        return {
            "level": "Moderate",
            "advice": "You seem a bit tense. Try 10 deep breaths, stretch for a minute, or write your thoughts down 💪.",
            "actions": ["10 deep breaths", "Stretch briefly", "Jot thoughts in notes"]
        }
    else:
        return {
            "level": "Low",
            "advice": "Stress seems manageable 🌱. Keep your good habits — rest, hydrate, and stay positive.",
            "actions": ["Drink water", "5-min gratitude note", "Plan a mini reward"]
        }

# ── Routes ───────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    text = request.form.get("message", "").strip()
    if not text:
        return jsonify({"reply": "Could you share that again?", "type": "chat"})

    add_context("User", text)

    # Stress flow
    if state["stage"] == "stress":
        state["answers"].append(text)
        if len(state["answers"]) >= 3:
            score = score_stress(state["answers"])
            rec = stress_recommendation(score)
            state.update({"stage": None, "answers": [], "offered_stress": False})
            reply = (
                f"🧘 Stress Level: {rec['level']}\n{rec['advice']}\n"
                f"Suggested actions:\n• {rec['actions'][0]}\n• {rec['actions'][1]}\n• {rec['actions'][2]}"
            )
            add_context("MindMate", reply)
            return jsonify({"reply": reply, "type": "result"})
        else:
            q = STRESS_QUESTIONS[len(state["answers"])]
            add_context("MindMate", q)
            return jsonify({"reply": q, "type": "stress"})

    # Offer stress test
    if any(k in text.lower() for k in ["not good", "sad", "anxious", "stressed", "tensed", "depressed"]):
        reply = "It sounds tough 😔. Want to take a quick 3-question stress check?"
        state["offered_stress"] = True
        add_context("MindMate", reply)
        return jsonify({"reply": reply, "type": "offer_test"})

    # Start stress test
    if state["offered_stress"] and any(k in text.lower() for k in ["yes", "sure", "ok", "start", "take test"]):
        q = STRESS_QUESTIONS[0]
        state.update({"stage": "stress", "answers": [], "offered_stress": False})
        add_context("MindMate", q)
        return jsonify({"reply": q, "type": "stress"})

    # Relaxation resources
    if any(k in text.lower() for k in ["relax", "calm", "breathe", "meditate", "anger control", "cool down"]):
        add_context("MindMate", RELAXATION_SNIPPET)
        return jsonify({"reply": RELAXATION_SNIPPET, "type": "resource"})

    # Emotion analysis + Gemini response
    emotion = get_emotion(text)
    reply = gemini_reply(text, emotion)
    add_context("MindMate", reply)
    return jsonify({"reply": reply, "type": "chat"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
