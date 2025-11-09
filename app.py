import os
import json
import random
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from emotion_model import get_emotion

# Load environment variables
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI setup
try:
    from openai import OpenAI
    oai_client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
except Exception:
    oai_client = None

# Flask app
app = Flask(__name__)

# Load resources from JSON
with open("resources.json", "r", encoding="utf-8") as f:
    RESOURCES = json.load(f)

# In-memory state for stress test
user_states = {}  # { user_id: {"stage":"stress","answers":[...]} }

# Stress test questions
STRESS_QUESTIONS = [
    "Do you often feel overwhelmed or tense? (Often / Sometimes / Rarely)",
    "Do you have trouble relaxing or sleeping? (Often / Sometimes / Rarely)",
    "Do you find it hard to focus on tasks? (Often / Sometimes / Rarely)"
]

# Fallback replies if GPT is unavailable
FALLBACK_RESPONSES = {
    "joy": ["That’s lovely to hear! What made your day brighter?", "I love that energy—want to share what went well?"],
    "love": ["Love brings warmth—hold onto that feeling 💚", "That sounds meaningful. Who or what inspired it?"],
    "sadness": ["I’m sorry it’s heavy right now 😔. Want to talk about what’s weighing on you?",
                "It’s okay to not be okay—what would feel supportive right now?"],
    "fear": ["That sounds unsettling. Let’s slow down—what’s the biggest worry?",
             "You’re not alone. Would grounding or breathing help?"],
    "anger": ["It’s valid to feel angry 😤. What triggered it?", "Let’s unpack it—what might help release that tension?"],
    "neutral": ["I’m here—what’s been on your mind today?", "Tell me more; I’m listening."]
}


def gpt_reply(user_text: str, emotion: str) -> str:
    """Generate empathetic GPT reply or fallback."""
    if not oai_client:
        return random.choice(FALLBACK_RESPONSES.get(emotion, FALLBACK_RESPONSES["neutral"]))

    # Detect crisis words
    crisis_terms = ["suicide", "kill myself", "end my life", "self harm", "hurt myself"]
    if any(term in user_text.lower() for term in crisis_terms):
        return (
            "I’m really glad you told me. Your safety matters. "
            "If you’re in immediate danger, please call your local emergency number (112 in India) "
            "or reach out to a trusted person right now. You’re not alone."
        )

    system_prompt = (
        "You are MindMate, a warm, youth-focused mental wellness companion. "
        "Goals: (1) respond in 1–3 short sentences, (2) be empathetic and validating, "
        "(3) suggest small helpful actions (breathing, reflection, hydration, grounding), "
        "(4) end with a gentle follow-up question. "
        "Avoid medical advice or diagnoses. Keep tone supportive and safe."
    )

    user_context = f"Detected emotion: {emotion}. User said: {user_text}\nRespond now."

    try:
        response = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_context}],
            temperature=0.7,
            max_tokens=120,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return random.choice(FALLBACK_RESPONSES.get(emotion, FALLBACK_RESPONSES["neutral"]))


def score_stress(answers):
    """Score user’s stress level from responses."""
    total = 0
    for ans in answers:
        ans = ans.lower()
        if "often" in ans:
            total += 3
        elif "sometimes" in ans:
            total += 2
        elif "rarely" in ans:
            total += 1
    return total


def stress_recommendation(score):
    """Return stress recommendation and resources."""
    if score >= 7:
        key = "high_stress"
        level = "High"
        advice = ("Your stress seems high 😟. Try a 3–5 minute breathing routine "
                  "(inhale 4s, hold 4s, exhale 6s), a short walk, or journaling. "
                  "If you feel unsafe, reach local help (112 India) or talk to someone you trust.")
    elif score >= 4:
        key = "moderate_stress"
        level = "Moderate"
        advice = ("There are signs of stress 😕. Small routines help—5-minute stretch, "
                  "10 deep breaths, and finishing a small task to regain momentum.")
    else:
        key = "low_stress"
        level = "Low"
        advice = ("Good job 🌿 — stress looks manageable. Keep your sleep, hydration, and positivity.")

    rec = {
        "level": level,
        "advice": advice,
        "actions": [
            "Take a 5-minute breathing break",
            "Drink water and stretch",
            "Write one positive thing you did today"
        ],
        "resources": RESOURCES.get(key, [])
    }
    return rec


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_id = "default_user"
    text = request.form.get("message", "").strip().lower()

    if not text:
        return jsonify({"reply": "Could you say that again?", "type": "chat"})

    # 🌿 Ongoing stress test
    if user_id in user_states and user_states[user_id]["stage"] == "stress":
        answers = user_states[user_id]["answers"]
        answers.append(text)
        if len(answers) >= 3:
            score = score_stress(answers)
            rec = stress_recommendation(score)
            del user_states[user_id]

            res_text = "\n\nHere are some things that might help:\n"
            for r in rec["resources"]:
                res_text += f"• [{r['title']}]({r['link']}) ({r['type']})\n"

            reply = (
                f"Stress level: {rec['level']}\n"
                f"{rec['advice']}\n"
                f"Suggested actions: • {rec['actions'][0]} • {rec['actions'][1]} • {rec['actions'][2]}"
                f"{res_text}"
            )
            return jsonify({"reply": reply, "type": "result"})
        else:
            q_idx = len(answers)
            return jsonify({"reply": STRESS_QUESTIONS[q_idx], "type": "stress"})

    # 🌿 Relaxation / help triggers
    relax_triggers = [
        "relaxation", "relax", "techniques", "calm", "help me relax",
        "breathing", "stress relief", "meditation", "relax guide"
    ]
    if any(word in text for word in relax_triggers):
        resources = RESOURCES.get("moderate_stress", [])
        res_text = "Here are some relaxation techniques and resources 🌿:\n\n"
        for r in resources:
            res_text += f"• [{r['title']}]({r['link']}) ({r['type']})\n"
        res_text += "\nWould you like to take a quick 3-question stress check?"
        return jsonify({"reply": res_text, "type": "offer_test"})

    # 🌿 Intent to start stress test (detect 'yes', 'start', etc.)
    start_test_triggers = [
        "yes", "yeah", "start", "ok start", "i want to take", "begin", "take stress test", "sure"
    ]
    if any(word in text for word in start_test_triggers):
        user_states[user_id] = {"stage": "stress", "answers": []}
        return jsonify({"reply": STRESS_QUESTIONS[0], "type": "stress"})

    # 🌿 Emotion detection
    emotion = get_emotion(text)

    # 🌿 If user is feeling bad — offer stress check
    bad_mood_words = ["not good", "sad", "down", "depressed", "anxious", "angry", "scared", "overwhelmed", "stressed"]
    if any(word in text for word in bad_mood_words):
        return jsonify({"reply": "It sounds tough 😔. Want to take a quick 3-question stress check?", "type": "offer_test"})

    # 🌿 Otherwise: normal empathetic GPT response
    reply = gpt_reply(text, emotion)
    return jsonify({"reply": reply, "type": "chat"})



if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

