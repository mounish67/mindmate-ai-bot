import os
import random
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from emotion_model import get_emotion

# ── Setup ──────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")           # optional: OpenAI GPT
HF_TOKEN   = os.getenv("HF_TOKEN", "")             # optional: Hugging Face (better rate limits)

app = Flask(__name__)

# In-memory state (single-user demo). For multiuser, swap to Flask session IDs.
# state = {
#   "stage": None|"stress",
#   "answers": [],
#   "offered_stress": False,
#   "context": ["User: ...", "MindMate: ...", ...]  # rolling window
# }
state = {"stage": None, "answers": [], "offered_stress": False, "context": []}

STRESS_QUESTIONS = [
    "Do you often feel overwhelmed or tense? (Often / Sometimes / Rarely)",
    "Do you have trouble relaxing or sleeping? (Often / Sometimes / Rarely)",
    "Do you find it hard to focus on tasks? (Often / Sometimes / Rarely)"
]

# Friendly resources we can offer proactively
RELAXATION_SNIPPET = (
    "Here are quick relaxation tools 🌿:\n"
    "• 10-Minute Guided Relaxation (YouTube): https://www.youtube.com/watch?v=inpok4MKVLM\n"
    "• Gratitude Journaling Tips (Article): https://positivepsychology.com/gratitude-journal/\n"
    "• Positive Thinking Audio (Spotify): https://open.spotify.com/track/6dGnYIeXmHdcikdzNNDMm2\n"
)

# Empathetic fallbacks (used only if all models fail)
FALLBACK_RESPONSES = {
    "joy": [
        "That’s wonderful 😊 What made your day brighter?",
        "Love that energy — what went well?"
    ],
    "love": [
        "That sounds meaningful 💚 What sparked that feeling?",
        "Hold onto that warmth — want to share more?"
    ],
    "sadness": [
        "I’m sorry it feels heavy. What’s weighing on you most?",
        "It’s okay to not be okay — I’m here with you."
    ],
    "fear": [
        "That sounds unsettling. Let’s take one slow breath together.",
        "You’re not alone. What’s the biggest worry right now?"
    ],
    "anger": [
        "It’s valid to feel angry. What triggered it?",
        "Let’s unpack it gently — what might help release it a bit?"
    ],
    "surprise": [
        "Whoa — unexpected! How are you feeling about it now?",
        "Surprises can throw us off — was it good or tough?"
    ],
    "neutral": [
        "I’m here — what’s been on your mind today?",
        "Tell me more; I’m listening."
    ]
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def add_context(role: str, text: str, keep_last: int = 8):
    """Append a line to rolling context."""
    state["context"].append(f"{role}: {text}")
    if len(state["context"]) > keep_last:
        state["context"] = state["context"][-keep_last:]


def recent_context_text() -> str:
    """Return a compact context string for models."""
    return "\n".join(state["context"][-8:])


# ── OpenAI GPT reply (returns None on any error so we can fallback) ────────────
def gpt_reply(user_text: str, emotion: str) -> str | None:
    if not OPENAI_KEY:
        return None

    # Basic crisis safety
    crisis_terms = ["suicide", "kill myself", "end my life", "self harm", "cut myself", "hurt myself"]
    if any(term in user_text.lower() for term in crisis_terms):
        return (
            "I’m really glad you told me. Your safety matters. "
            "If you’re in immediate danger, please call 112 (India) or reach out to a trusted person nearby 💛."
        )

    system_prompt = (
        "You are MindMate, a warm, youth-focused mental wellness companion. "
        "Be natural and empathetic. Use 1–3 short sentences. Validate feelings, suggest one tiny action "
        "(breathing, grounding, journaling, short walk), and end with a gentle question. "
        "Avoid clinical/diagnostic language."
    )

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{recent_context_text()}\n\nDetected emotion: {emotion}\nUser: {user_text}"}
        ],
        "temperature": 0.85,
        "max_tokens": 180
    }

    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}",
                     "Content-Type": "application/json"},
            json=payload,
            timeout=20
        )
        if r.status_code != 200:
            print(f"⚠️ GPT ERROR {r.status_code}: {r.text}")
            return None
        data = r.json()
        if "choices" in data and data["choices"]:
            content = data["choices"][0].get("message", {}).get("content", "")
            return content.strip() if content else None
        print(f"⚠️ GPT unexpected payload: {data}")
        return None
    except Exception as e:
        print(f"⚠️ GPT ERROR: {e}")
        return None


# ── Improved Hugging Face fallback (empathetic + context + tuned params) ──────
def hf_fallback_reply(user_text: str, emotion: str) -> str:
    """
    Use a conversational HF model with a warm system-style instruction and recent context,
    so replies are longer, empathetic, and feel human (not generic).
    """
    print("⚙️ Using Hugging Face fallback reply...")
    system_prompt = (
        "You are MindMate, a friendly mental wellness companion. "
        "Speak in 1–3 short, warm sentences. Acknowledge feelings, offer one simple coping tip "
        "(e.g., slow breathing, short walk, grounding, journaling), and end with a gentle question. "
        "Sound human and supportive. Avoid clinical terms."
    )

    # Prefer BlenderBot 400M Distill for coherent short replies
    model_url = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    payload = {
        "inputs": f"{system_prompt}\n\nRecent conversation:\n{recent_context_text()}\n\nUser ({emotion}): {user_text}",
        "parameters": {
            "temperature": 0.9,
            "repetition_penalty": 1.12,
            "max_new_tokens": 120,
            "top_p": 0.95,
            "return_full_text": False
        }
    }

    try:
        res = requests.post(model_url, headers=headers, json=payload, timeout=20)
        # HF returns either dict or list; handle both
        data = res.json()
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()

        # Some models return {'error': '...loading...'} on first call; retry once lightly
        if isinstance(data, dict) and "error" in data:
            print(f"ℹ️ HF warmup: {data['error']}")
            res = requests.post(model_url, headers=headers, json=payload, timeout=20)
            data = res.json()
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"].strip()
            if isinstance(data, list) and data and "generated_text" in data[0]:
                return data[0]["generated_text"].strip()

        print(f"⚠️ HF unexpected payload: {data}")
    except Exception as e:
        print(f"⚠️ HF ERROR: {e}")

    # Last-chance empathetic fallback
    return random.choice(FALLBACK_RESPONSES.get(emotion, FALLBACK_RESPONSES["neutral"]))


# ── Stress Test helpers ───────────────────────────────────────────────────────
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
                "Your stress seems high 😟. Try 4-4-6 breathing for 3–5 minutes, step outside for fresh air, "
                "and jot down what feels heaviest. If it feels unsafe, please reach someone you trust."
            ),
            "actions": [
                "Do box-breathing (inhale 4s, hold 4s, exhale 6s)",
                "Short walk or gentle stretch",
                "Text/call a trusted friend"
            ]
        }
    elif score >= 4:
        return {
            "level": "Moderate",
            "advice": (
                "There are signs of stress. Try 10 slow breaths, a 5-minute stretch, and finish one tiny task to regain momentum."
            ),
            "actions": [
                "10 slow breaths",
                "5-minute stretch",
                "Write 3 thoughts to let go"
            ]
        }
    else:
        return {
            "level": "Low",
            "advice": (
                "Nice — stress looks manageable. Keep your good habits: sleep, hydration, and light movement."
            ),
            "actions": [
                "5-minute gratitude note",
                "Drink water + short walk",
                "Plan one small enjoyable thing"
            ]
        }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    text = request.form.get("message", "").strip()
    if not text:
        return jsonify({"reply": "Could you share that again?", "type": "chat"})

    # Reset command
    if any(k in text.lower() for k in ["restart", "reset", "clear chat", "start over"]):
        state.update({"stage": None, "answers": [], "offered_stress": False, "context": []})
        return jsonify({"reply": "Chat cleared 🌱. Hey there! How are you feeling today?", "type": "chat"})

    # Ongoing stress test
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

    # Emotion detection
    emotion = get_emotion(text)

    # Offer stress test on negative cues
    if any(k in text.lower() for k in [
        "not good","sad","down","depressed","anxious","stressed","overwhelmed","angry","tensed","tired","worried"
    ]):
        state["offered_stress"] = True
        offer = "It sounds tough 😔. Want to take a quick 3-question stress check?"
        add_context("User", text); add_context("MindMate", offer)
        return jsonify({"reply": offer, "type": "offer_test"})

    # Accept stress test if user agrees after an offer
    if state.get("offered_stress") and any(k in text.lower() for k in ["yes","sure","ok","start","take test","yeah","yup"]):
        state.update({"stage": "stress", "answers": [], "offered_stress": False})
        q = STRESS_QUESTIONS[0]
        add_context("User", text); add_context("MindMate", q)
        return jsonify({"reply": q, "type": "stress"})

    # Quick relaxation keyword path
    if any(k in text.lower() for k in ["relax","calm","breathe","meditate","anger control","cool down","stress relief"]):
        add_context("User", text); add_context("MindMate", RELAXATION_SNIPPET)
        return jsonify({"reply": RELAXATION_SNIPPET, "type": "resource"})

    # Otherwise generate reply (GPT → HF fallback), with rolling context
    add_context("User", text)
    reply = gpt_reply(text, emotion)
    if not reply:
        reply = hf_fallback_reply(text, emotion)
    add_context("MindMate", reply)

    return jsonify({"reply": reply, "type": "chat"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # In production on Render, Gunicorn runs this app; this line is for local runs.
    app.run(host="0.0.0.0", port=port)
