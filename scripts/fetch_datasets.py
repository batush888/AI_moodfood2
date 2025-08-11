import os
import json
import requests

def load_dataset(source: str, cache_path: str = "data/intent_dataset_cached.json"):
    """
    Loads dataset from a URL or local file.
    
    Args:
        source (str): Can be a local path (e.g., 'data/intent_dataset.json')
                      or an online URL (e.g., 'https://example.com/dataset.json').
        cache_path (str): Path to store a local cached copy if loaded from the internet.

    Returns:
        list: Dataset as a list of dicts.
    """
    # If it's a URL
    if source.startswith("http://") or source.startswith("https://"):
        print(f"Fetching dataset from {source} ...")
        response = requests.get(source)
        response.raise_for_status()
        dataset = response.json()

        # Cache it locally for offline use
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)
        print(f"Dataset cached at {cache_path}")

    # If it's a local path
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"Local dataset not found at {source}")
        with open(source, "r", encoding="utf-8") as f:
            dataset = json.load(f)

    return dataset