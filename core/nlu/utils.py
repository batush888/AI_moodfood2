# core/nlu/utils.py
import json
import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.label_utils import load_taxonomy

def validate_intent_dataset(dataset_path: str, taxonomy_path: str):
    """
    Validates the dataset against the normalized taxonomy.

    Args:
        dataset_path (str): Path to intent_dataset.json
        taxonomy_path (str): Path to mood_food_taxonomy.json
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not os.path.exists(taxonomy_path):
        raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_path}")

    # Use the normalized taxonomy from label_utils
    taxonomy = load_taxonomy()
    valid_labels = set(taxonomy.keys())

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    missing_labels = set()
    valid_entries = 0
    total_labels = 0
    
    for idx, entry in enumerate(dataset):
        if "labels" not in entry or not isinstance(entry["labels"], list):
            raise ValueError(f"Entry {idx} has no valid 'labels' list.")

        entry_has_valid_labels = False
        for label in entry["labels"]:
            total_labels += 1
            if label not in valid_labels:
                missing_labels.add(label)
            else:
                entry_has_valid_labels = True
        
        if entry_has_valid_labels:
            valid_entries += 1

    # Calculate coverage
    coverage = (valid_entries / len(dataset)) * 100 if dataset else 0
    
    print(f"📊 Dataset validation summary:")
    print(f"   • Total entries: {len(dataset)}")
    print(f"   • Entries with valid labels: {valid_entries}")
    print(f"   • Coverage: {coverage:.1f}%")
    print(f"   • Total labels: {total_labels}")
    print(f"   • Missing labels: {len(missing_labels)}")
    
    if missing_labels:
        print(f"   ⚠️  Missing labels (first 10): {sorted(list(missing_labels))[:10]}")
        print(f"   💡 Training will proceed with available common labels")
    
    if coverage < 50:
        print(f"   ⚠️  WARNING: Low coverage ({coverage:.1f}%). Consider expanding taxonomy.")
    
    print(f"✅ Dataset validation completed. Training can proceed.")