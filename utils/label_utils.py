# AI_moodfood2/utils/label_utils.py

import json
import os
from pathlib import Path
from typing import Dict, List, Union

# ------------------
# File paths
# ------------------
BASE_DIR = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = BASE_DIR / "data" / "taxonomy" / "mood_food_taxonomy.json"
DATASET_PATH = BASE_DIR / "data" / "intent_dataset.json"

# ------------------
# Loaders
# ------------------
def load_taxonomy(path: Path = TAXONOMY_PATH) -> Dict[str, Dict]:
    """
    Loads the mood-food taxonomy from JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_labels_from_dataset(path: Path = DATASET_PATH) -> List[str]:
    """
    Load unique labels from the dataset file.
    """
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    labels = set()
    for sample in dataset:
        for label in sample.get("labels", []):
            labels.add(label)

    return sorted(labels)

# ------------------
# Mapping creators
# ------------------
def get_label_mappings(labels: List[str]) -> Dict[str, int]:
    """
    Creates a label-to-ID mapping from a list of label names.
    """
    return {label: idx for idx, label in enumerate(labels)}

def get_reverse_label_mappings(label_to_id: Dict[str, int]) -> Dict[int, str]:
    """
    Creates an ID-to-label mapping from a label-to-ID dictionary.
    """
    return {v: k for k, v in label_to_id.items()}

# ------------------
# Normalization helpers
# ------------------
def normalize_labels(
    labels: Union[List[str], str],
    label_to_id: Dict[str, int]
) -> List[int]:
    """
    Converts label names to their corresponding integer IDs.
    """
    if isinstance(labels, str):
        labels = [labels]
    return [label_to_id[label] for label in labels if label in label_to_id]

def denormalize_ids(
    ids: Union[List[int], int],
    id_to_label: Dict[int, str]
) -> List[str]:
    """
    Converts label IDs back to their corresponding label names.
    """
    if isinstance(ids, int):
        ids = [ids]
    return [id_to_label[idx] for idx in ids if idx in id_to_label]

# ------------------
# Preload both systems
# ------------------

# 1. From taxonomy
TAXONOMY = load_taxonomy()
TAXONOMY_LABELS = sorted(TAXONOMY.keys())
TAXONOMY_LABEL_TO_ID = get_label_mappings(TAXONOMY_LABELS)
TAXONOMY_ID_TO_LABEL = get_reverse_label_mappings(TAXONOMY_LABEL_TO_ID)

# 2. From dataset (auto-extracted)
DATASET_LABELS = load_labels_from_dataset()
DATASET_LABEL_TO_ID = get_label_mappings(DATASET_LABELS)
DATASET_ID_TO_LABEL = get_reverse_label_mappings(DATASET_LABEL_TO_ID)

# ------------------
# Main interface
# ------------------
if __name__ == "__main__":
    print("📂 Taxonomy labels:", TAXONOMY_LABELS)
    print("📂 Dataset labels:", DATASET_LABELS)
    print("TAXONOMY_LABEL_TO_ID:", TAXONOMY_LABEL_TO_ID)
    print("DATASET_LABEL_TO_ID:", DATASET_LABEL_TO_ID)