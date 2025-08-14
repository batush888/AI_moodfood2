# core/nlu/normalize_dataset.py
import json
import re
from pathlib import Path
from typing import List, Dict, Optional

# Synonym mapping for domain-specific consistency
SYNONYM_MAP = {
    # Comfort Emotional
    "goal_comfort": "comfort",
    "emotion_sad": "sad",
    "sensory_soothing": "soothing",
    "temperature_warm": "comfort",   # warm food is usually comfort food
    "emotion_gloomy": "sad",         # mood proxy

    # Energy Vitality
    "goal_energy": "need_a_boost",   # matches "need a boost" descriptor
    "activity_gym": "need_a_boost",  # gym implies energy need

    # Romantic Dinner
    "occasion_romantic": "romantic",
    "meal_dinner": "romantic",       # dinner context in romantic taxonomy

    # Health / Light
    "goal_light": "light",
    "goal_healthy": "healthy",
    "temperature_cold": "refreshing",  # not in taxonomy, map to "healthy/light" group
    "sensory_refreshing": "light",     # closest fit in "healthy/light"

    # Misc - Needs decisions
    "meal_lunch": "light",             # lunch could go in healthy/light
    "weather_rainy": "comfort",        # rainy → comfort food
    "health_illness": "soothing",      # sick → soothing food
    "food_smoothie": "smoothie",       # exists in ENERGY_VITALITY foods list
    "time_breakfast": "light",         # breakfast is often light/healthy
    "goal_quick": "light",             # quick meals → healthy/light
    "social_alone": "comfort",         # alone eating → comfort
    "goal_simple": "comfort",          # simple → comfort meals
    "social_group": "romantic",        # group dinner → special night
    "weather_hot": "light"             # hot → light/cold meals
}

def normalize_label_string(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    # Map synonyms
    return SYNONYM_MAP.get(s, s)

def normalize_labels(labels: List[str]) -> List[str]:
    """
    Normalize a list of labels: lowercase, underscores, synonym mapping.
    Removes duplicates while preserving order.
    """
    normalized = []
    for l in labels or []:
        nl = normalize_label_string(l)
        if nl:
            normalized.append(nl)
    seen = set()
    out = []
    for item in normalized:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out

def normalize_sample(sample: Dict) -> Optional[Dict]:
    """
    Normalize one dataset entry: ensure text and normalized labels.
    """
    text = sample.get("text", "").strip()
    if not text:
        return None

    raw_labels = (
        sample.get("labels")
        or sample.get("label")
        or sample.get("intents")
        or []
    )
    if isinstance(raw_labels, str):
        raw_labels = [raw_labels]

    labels = normalize_labels([l for l in raw_labels if isinstance(l, str) and l.strip()])
    return {"text": text, "labels": labels}

def load_and_normalize_dataset(input_path: str) -> List[Dict]:
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    normalized = []
    seen = set()
    for entry in raw_data:
        norm = normalize_sample(entry)
        if norm is None:
            continue
        key = (norm["text"], tuple(sorted(norm["labels"])))
        if key not in seen:
            normalized.append(norm)
            seen.add(key)
    return normalized

def save_dataset(data: List[Dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved normalized dataset with {len(data)} samples to: {output_path}")

if __name__ == "__main__":
    INPUT_PATH = "data/intent_dataset.json"
    OUTPUT_PATH = "data/intent_dataset_normalized.json"

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    normalized_data = load_and_normalize_dataset(INPUT_PATH)
    save_dataset(normalized_data, OUTPUT_PATH)