# core/taxonomy/loader.py

import json
from pathlib import Path

def load_taxonomy(path: str = "data/taxonomy/mood_food_taxonomy.json"):
    with open(path, "r") as f:
        return json.load(f)