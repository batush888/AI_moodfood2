# core/nlu/utils.py
import json
import os

def validate_intent_dataset(dataset_path: str, taxonomy_path: str):
    """
    Validates the dataset against the taxonomy.

    Args:
        dataset_path (str): Path to intent_dataset.json
        taxonomy_path (str): Path to mood_food_taxonomy.json
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not os.path.exists(taxonomy_path):
        raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_path}")

    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    valid_labels = set(taxonomy.keys())

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    for idx, entry in enumerate(dataset):
        if "labels" not in entry or not isinstance(entry["labels"], list):
            raise ValueError(f"Entry {idx} has no valid 'labels' list.")

        for label in entry["labels"]:
            if label not in valid_labels:
                raise ValueError(
                    f"Invalid label '{label}' in entry {idx}. Must be one of {sorted(valid_labels)}"
                )

    print(f"✅ Dataset validation passed. {len(dataset)} entries checked.")