# nlu/entity_extraction.py

import re
from typing import Dict, List, Tuple

# Base dictionaries (you can externalize / load from JSON)
ENTITY_TYPES = {
    "DIETARY_RESTRICTIONS": ["vegetarian", "vegan", "gluten-free", "dairy-free", "keto", "paleo", "halal", "kosher"],
    "ALLERGIES": ["nuts", "shellfish", "eggs", "soy", "dairy", "peanuts", "gluten"],
    "CUISINE_PREFERENCES": ["italian", "chinese", "mexican", "indian", "thai", "japanese", "french", "korean"],
    "MEAL_TYPE": ["breakfast", "lunch", "dinner", "snack", "dessert"],
    "TIME_CONTEXT": ["morning", "afternoon", "evening", "late night", "late_night"],
    "SOCIAL_CONTEXT": ["alone", "with friends", "family dinner", "date night", "couple", "group"],
    "INTENSITY_MODIFIERS": ["really", "desperately", "slightly", "very", "extremely", "kind of", "a bit"],
    "NEGATION": ["not", "don't", "avoid", "without", "except", "no"]
}

# Precompile regexes for performance
def _build_pattern(words: List[str]) -> re.Pattern:
    escaped = [re.escape(w) for w in sorted(words, key=lambda x: -len(x))]
    pattern = r'\b(' + '|'.join(escaped) + r')\b'
    return re.compile(pattern, flags=re.IGNORECASE)

PATTERNS = {k: _build_pattern(v) for k, v in ENTITY_TYPES.items()}


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract entities from freeform text using dictionary/regex matching.
    Returns a dict of entity-type -> list of values found.
    """
    found: Dict[str, List[str]] = {k: [] for k in ENTITY_TYPES.keys()}

    normalized = text.lower()

    for entity_type, pattern in PATTERNS.items():
        for match in pattern.finditer(normalized):
            value = match.group(1).strip()
            # Normalize some variants (e.g., "late night" -> "late_night")
            if entity_type == "TIME_CONTEXT":
                value = value.replace(" ", "_")
            if entity_type == "SOCIAL_CONTEXT":
                value = value.replace(" ", "_")
            found[entity_type].append(value)

    # Post-process negations: if a negation appears near another entity, mark it
    negations = found.get("NEGATION", [])
    if negations:
        # simplistic: if 'not' appears, flag next few words as negated
        found["NEGATED"] = _identify_negated_phrases(text)

    return found


def _identify_negated_phrases(text: str, window: int = 3) -> List[str]:
    """
    Very simple negation detection: if a negation word is followed by a noun/adjective in the next `window` tokens,
    return those words as negated.
    """
    tokens = text.lower().split()
    negated = []
    negation_terms = ENTITY_TYPES["NEGATION"]
    for i, tok in enumerate(tokens):
        if tok in negation_terms:
            for j in range(1, window + 1):
                if i + j < len(tokens):
                    negated.append(tokens[i + j])
    return negated