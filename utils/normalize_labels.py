import json
from typing import List, Dict, Any
import os
from collections import defaultdict

def load_taxonomy(taxonomy_path: str) -> Dict[str, Any]:
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")

def map_single_to_multi_labels(example: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all labels are under 'labels' key as a list"""
    if "label" in example:
        example["labels"] = [normalize_label(example["label"])]
        del example["label"]
    else:
        example["labels"] = [normalize_label(label) for label in example.get("labels", [])]
    return example

def expand_labels_with_synonyms(example: Dict[str, Any], taxonomy: Dict[str, Any]) -> List[str]:
    """Add any known synonym or grouped label for each label"""
    expanded = set(example["labels"])
    for label in example["labels"]:
        for group_key, group_value in taxonomy.items():
            if label == normalize_label(group_key) or label in map(normalize_label, group_value.get("related_labels", [])):
                expanded.add(normalize_label(group_key))
                for related in group_value.get("related_labels", []):
                    expanded.add(normalize_label(related))
    return list(expanded)

def process_dataset(dataset_path: str, taxonomy_path: str, output_path: str):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    taxonomy = load_taxonomy(taxonomy_path)

    processed_dataset = []
    for example in dataset:
        example = map_single_to_multi_labels(example)
        example["labels"] = expand_labels_with_synonyms(example, taxonomy)
        processed_dataset.append(example)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_dataset, f, indent=4, ensure_ascii=False)

    print(f"✅ Dataset processed and saved to {output_path}")

# Usage
if __name__ == "__main__":
    DATASET_PATH = "data/intent_dataset.json"
    TAXONOMY_PATH = "data/taxonomy/mood_food_taxonomy.json"
    OUTPUT_PATH = "data/intent_dataset.normalized.json"

    process_dataset(DATASET_PATH, TAXONOMY_PATH, OUTPUT_PATH)


