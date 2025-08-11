# nlu/intent_classification.py

from typing import List, Dict
import re

# Simple keyword-to-intent mapping fallback; later swap with a learned classifier
INTENT_KEYWORDS = {
    "COMFORT_EMOTIONAL": ["comfort", "stressed", "sad", "nostalgia", "celebrate", "feel bad", "cozy", "warm me up"],
    "ENERGY_VITALITY": ["energy", "tired", "sluggish", "wake me up", "brain", "fuel", "boost"],
    "WEATHER_SEASONAL": ["cold", "hot", "rainy", "chilly", "heat", "summer", "winter"],
    "TEXTURE_SENSORY": ["crunchy", "creamy", "smooth", "chewy", "crispy"],
    "FLAVOR_PROFILE": ["spicy", "sweet", "salty", "umami", "sour", "bitter", "tangy"],
    "OCCASION_SOCIAL": ["quick", "romantic", "party", "family", "date", "share"],
    "HEALTH_WELLNESS": ["light", "healthy", "detox", "recovery", "hangover"]
}

INTENT_CONF_THRESHOLDS = {
    "COMFORT_EMOTIONAL": 0.6,
    "ENERGY_VITALITY": 0.6,
    "WEATHER_SEASONAL": 0.7,
    "TEXTURE_SENSORY": 0.8,
    "FLAVOR_PROFILE": 0.7,
    "OCCASION_SOCIAL": 0.6,
    "HEALTH_WELLNESS": 0.6
}


def classify_intent(text: str) -> Dict:
    """
    Naive intent classifier using keyword matching.
    Returns primary intent, secondary intents, and a confidence proxy.
    """
    normalized = text.lower()
    scores: Dict[str, float] = {}

    for intent, keywords in INTENT_KEYWORDS.items():
        match_count = 0
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', normalized):
                match_count += 1
        # Simple scoring: proportional to matches over number of keywords
        scores[intent] = match_count / max(1, len(keywords))

    # Sort intents
    sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary_intent, primary_score = sorted_intents[0]
    secondary_intents = [i for i, s in sorted_intents[1:3] if s > 0]

    # Confidence threshold logic
    threshold = INTENT_CONF_THRESHOLDS.get(primary_intent, 0.5)
    confidence = primary_score
    if primary_score < threshold:
        confidence = primary_score * 0.5  # lower confidence if below threshold

    return {
        "primary_intent": primary_intent,
        "secondary_intents": secondary_intents,
        "confidence": round(confidence, 3),
        "raw_scores": scores
    }

TRAINING_EXAMPLES = [
    {
        "text": "I'm feeling really stressed and want something comforting",
        "intent": "COMFORT_EMOTIONAL",
        "entities": [
            {"entity": "INTENSITY_MODIFIERS", "value": "really", "start": 12, "end": 18},
            {"entity": "EMOTION", "value": "stressed", "start": 19, "end": 27}
        ],
        "mood_intensity": 0.9
    },
    {
        "text": "It's cold outside and I need something warm and spicy",
        "intent": "WEATHER_SEASONAL",
        "entities": [
            {"entity": "WEATHER", "value": "cold", "start": 5, "end": 9},
            {"entity": "TEXTURE_PREFERENCE", "value": "warm", "start": 43, "end": 47},
            {"entity": "FLAVOR_PROFILE", "value": "spicy", "start": 52, "end": 57}
        ],
        "mood_intensity": 0.7
    }
]