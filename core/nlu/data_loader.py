import json
from pathlib import Path

def load_taxonomy(taxonomy_path):
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_dataset(dataset_path):
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
    taxonomy_path = Path("data/taxonomy/mood_food_taxonomy.json")
    dataset_path = Path("data/intent_dataset.json")

    taxonomy = load_taxonomy(taxonomy_path)
    dataset = load_dataset(dataset_path)

    taxonomy = validate_and_fix_labels(dataset, taxonomy, auto_add=True)

    # Save updated taxonomy if modified
    with open(taxonomy_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=4, ensure_ascii=False)