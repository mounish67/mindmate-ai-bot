import os
import random
import uuid
from datetime import timedelta
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from emotion_model import get_emotion

# --- setup ---
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI (new SDK)
try:
    from openai import OpenAI
    oai_client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
except Exception:
    oai_client = None

app = Flask(__name__)
app.secret_key = "mindmate_secret_key"
app.permanent_session_lifetime = timedelta(hours=1)

# Store user-specific states
user_states = {}  # { user_id: {"context":[], "stage":None, "answers":[]} }

STRESS_QUESTIONS = [
    "Do you often feel overwhelmed or tense? (Often / Sometimes / Rarely)",
    "Do you have trouble relaxing or sleeping? (Often / Sometimes / Rarely)",
    "Do you find it hard to focus on tasks? (Often / Sometimes / Rarely)"
]

FALLBACK_RESPONSES = {
    "joy": [
        "That’s lovely to hear! What made your day brighter?",
        "I love that energy—want to share what went well?"
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
        "That sounds unsettling. Let’s slow down—what’s worrying you most?",
        "You’re not alone. Would grounding or breathing help?"
    ],
    "anger": [
        "It’s valid to feel angry. What triggered it?",
        "Let’s unpack that anger gently—what would help release it?"
    ],
    "surprise": [
        "Whoa—unexpected! How are you feeling about it now?",
        "Surprises can throw us off—was it good or tough?"
    ],
    "neutral": [
        "I’m here—what’s been on your mind today?",
        "Tell me more; I’m listening."
    ]
}

def gpt_reply(user_text: str, emotion: str) -> str:
    """
    Use ChatGPT to craft a short, empathetic, safe reply.
    Falls back to rule-based responses if API key is missing.
    """
    if not oai_client:
        return random.choice(FALLBACK_RESPONSES.get(emotion, FALLBACK_RESPONSES["neutral"]))

    crisis_terms = ["suicide","kill myself","end my life","self harm","cut myself","hurt myself"]
    if any(term in user_text.lower() for term in crisis_terms):
        return (
            "I’m really glad you told me. Your safety matters. "
            "If you’re in immediate danger, please call your local emergency number (e.g., 112 in India) "
            "or reach out to a trusted person nearby right now. You’re not alone."
        )

    system_prompt = (
        "You are MindMate, a warm, youth-focused mental wellness companion. "
        "Goals: (1) respond in 1–3 short sentences, (2) be empathetic and validating, "
        "(3) offer a gentle, practical next step (breathing, grounding, journaling, tiny action), "
        "and (4) end with one kind follow-up question. "
        "Never diagnose or claim medical authority. "
        "Keep a calm, hopeful tone. Use emojis sparingly (0–1). "
        "Special focus: if user mentions stress, anger, or anxiety, "
        "suggest relaxation, breathing, or journaling techniques naturally within response."
    )

    user_context = f"Detected emotion: {emotion}. User said: {user_text}"

    try:
        resp = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_context}
            ],
            temperature=0.7,
            max_tokens=120,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return random.choice(FALLBACK_RESPONSES.get(emotion, FALLBACK_RESPONSES["neutral"]))

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
                "Your stress seems high 😟. Consider a 3–5 minute breathing routine "
                "(inhale 4s, hold 4s, exhale 6s), short walk, and journaling your thoughts. "
                "If it feels unsafe, please reach out to emergency services (112 in India) or a trusted person."
            ),
            "actions": [
                "Try box-breathing (4-4-4-4) for 2 minutes",
                "Take a short walk",
                "Text or call a friend you trust"
            ]
        }
    elif score >= 4:
        return {
            "level": "Moderate",
            "advice": (
                "There are signs of stress. Small habits help: 10 deep breaths, 5-minute stretch, "
                "and write one thought you want to release."
            ),
            "actions": [
                "Take 10 deep breaths (inhale 4s, exhale 6s)",
                "5-minute body stretch",
                "Write 3 thoughts you want to let go"
            ]
        }
    else:
        return {
            "level": "Low",
            "advice": (
                "Nice—your stress seems manageable. Keep up your positive habits: rest, hydration, and small joys."
            ),
            "actions": [
                "Note 3 things you’re grateful for",
                "Drink some water and move a bit",
                "Do something you enjoy for 5 minutes"
            ]
        }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    # Identify user session
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
    user_id = session["user_id"]

    if user_id not in user_states:
        user_states[user_id] = {"context": [], "stage": None, "answers": []}

    text = request.form.get("message", "").strip()
    if not text:
        return jsonify({"reply": "Could you share that again?", "type": "chat"})

    # Handle chat reset
    if any(word in text.lower() for word in ["restart", "clear chat", "start over", "reset"]):
        user_states[user_id] = {"context": [], "stage": None, "answers": []}
        return jsonify({"reply": "Chat cleared 🌱. Hey there! How are you feeling today?", "type": "chat"})

    # Handle ongoing stress test
    if user_states[user_id]["stage"] == "stress":
        answers = user_states[user_id]["answers"]
        answers.append(text)
        if len(answers) >= 3:
            score = score_stress(answers)
            rec = stress_recommendation(score)
            user_states[user_id]["stage"] = None
            user_states[user_id]["answers"] = []
            reply = (
                f"Stress level: {rec['level']}\n"
                f"{rec['advice']}\n"
                f"Suggested actions:\n• {rec['actions'][0]}\n• {rec['actions'][1]}\n• {rec['actions'][2]}"
            )
            return jsonify({"reply": reply, "type": "result"})
        else:
            q_idx = len(answers)
            return jsonify({"reply": STRESS_QUESTIONS[q_idx], "type": "stress"})

    # Detect emotion and handle special cases
    emotion = get_emotion(text)

    if any(k in text.lower() for k in ["not good", "sad", "down", "depressed", "anxious", "angry", "overwhelmed", "tired", "moody"]):
        return jsonify({"reply": "It sounds tough 😔. Want to take a quick 3-question stress check?", "type": "offer_test"})

    if any(k in text.lower() for k in ["start test", "take test", "yes start", "ok start", "yes, start"]):
        user_states[user_id]["stage"] = "stress"
        user_states[user_id]["answers"] = []
        return jsonify({"reply": STRESS_QUESTIONS[0], "type": "stress"})

    # Maintain session context
    user_states[user_id]["context"].append(f"User: {text}")
    if len(user_states[user_id]["context"]) > 6:
        user_states[user_id]["context"] = user_states[user_id]["context"][-6:]

    context_text = "\n".join(user_states[user_id]["context"])
    reply = gpt_reply(f"Conversation so far:\n{context_text}\nUser latest: {text}", emotion)
    user_states[user_id]["context"].append(f"MindMate: {reply}")

    return jsonify({"reply": reply, "type": "chat"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
