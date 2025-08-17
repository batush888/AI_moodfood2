import json
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.label_utils import load_taxonomy, load_dataset_labels

def load_taxonomy_from_path(taxonomy_path):
    """Load taxonomy from a specific path (for backward compatibility)."""
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_dataset_from_path(dataset_path):
    """Load dataset from a specific path (for backward compatibility)."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)

def validate_and_fix_labels(dataset, taxonomy, auto_add=False):
    """
    Validate dataset labels against taxonomy.
    If labels are missing in taxonomy, log warning.
    Optionally auto-add them if auto_add=True.
    """
    taxonomy_labels = set(taxonomy.keys())
    new_labels = set()

    for sample in dataset:
        for label in sample["labels"]:
            if label not in taxonomy_labels:
                print(f"⚠️ Warning: '{label}' not found in taxonomy.")
                if auto_add:
                    taxonomy[label] = {"descriptors": [], "foods": []}
                    new_labels.add(label)

    if auto_add and new_labels:
        print(f"✅ Added {len(new_labels)} new labels to taxonomy: {new_labels}")

    return taxonomy

if __name__ == "__main__":
    print("🔍 Testing new reusable label functions...")
    
    # Use the new reusable functions
    taxonomy = load_taxonomy()
    dataset_labels = load_dataset_labels()
    
    print(f"✅ Loaded taxonomy with {len(taxonomy)} categories")
    print(f"✅ Loaded {len(dataset_labels)} unique dataset labels")
    
    # For validation, we need the full dataset
    dataset_path = Path("data/intent_dataset.json")
    dataset = load_dataset_from_path(dataset_path)
    
    taxonomy = validate_and_fix_labels(dataset, taxonomy, auto_add=True)

    # Save updated taxonomy if modified
    taxonomy_path = Path("data/taxonomy/mood_food_taxonomy.json")
    with open(taxonomy_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=4, ensure_ascii=False)