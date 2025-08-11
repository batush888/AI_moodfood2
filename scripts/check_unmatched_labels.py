# scripts/check_unmatched_labels.py
import json, os
import sys

# Add the parent directory to sys.path to resolve imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from core.nlu.normalize_dataset import normalize_labels
from scripts.train_intent_model import extract_all_sub_labels


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TAXONOMY_PATH = os.path.join(BASE_DIR, "..", "data", "taxonomy", "mood_food_taxonomy.json")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "intent_dataset.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

taxonomy = extract_all_sub_labels(TAXONOMY_PATH)

unmatched = {}
for i, item in enumerate(raw):
    labels = item.get("labels") or item.get("label") or item.get("intents") or []
    if isinstance(labels, str):
        labels = [labels]
    norm = normalize_labels(labels)
    for nl in norm:
        if nl not in taxonomy:
            unmatched.setdefault(nl, 0)
            unmatched[nl] += 1

print("Unmatched labels (normalized) and counts:")
for lbl, cnt in sorted(unmatched.items(), key=lambda x: -x[1]):
    print(lbl, cnt)