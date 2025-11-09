from transformers import pipeline

# Load once, return all scores so we can decide confidently
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# Simple keyword assist to correct short/ambiguous inputs
KEYWORDS = {
    "happy":"joy","glad":"joy","excited":"joy","awesome":"joy",
    "sad":"sadness","unhappy":"sadness","depressed":"sadness","cry":"sadness","low":"sadness","not good":"sadness","bad":"sadness","moody":"sadness",
    "angry":"anger","mad":"anger","furious":"anger",
    "scared":"fear","afraid":"fear","nervous":"fear","worried":"fear","anxious":"fear","anxiety":"fear",
    "love":"love","like":"love","care":"love",
    "shock":"surprise","wow":"surprise","unexpected":"surprise"
}

def get_emotion(text: str) -> str:
    t = text.lower()
    for k,v in KEYWORDS.items():
        if k in t:
            return v
    scores = emotion_classifier(text)[0]
    best = max(scores, key=lambda x: x["score"])
    return best["label"].lower()
