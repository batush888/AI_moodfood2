# core/nlu/intent_dataset.py
import os
import json
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import Dataset
from transformers import AutoTokenizer
from normalize_dataset import normalize_labels  # ✅ we already use this
from typing import List, Dict, Any, Optional, Set
from transformers.tokenization_utils import PreTrainedTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
import torch


# ✅ New helper function to get all taxonomy labels
def extract_all_sub_labels(taxonomy_path: str) -> Set[str]:
    """
    Extract all normalized labels from taxonomy.
    Supports dict-with-descriptors and list formats.
    """
    def normalize_label(s: str) -> str:
        return s.strip().lower().replace(" ", "_")

    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    label_set = set()
    for category, v in taxonomy.items():
        if isinstance(v, dict):
            descriptors = v.get("descriptors") or v.get("labels") or []
            if isinstance(descriptors, str):
                descriptors = [descriptors]
            for d in descriptors:
                label_set.add(normalize_label(d))
        elif isinstance(v, list):
            for item in v:
                label_set.add(normalize_label(item))
        elif isinstance(v, str):
            label_set.add(normalize_label(v))
    return label_set


class IntentDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        taxonomy_path: Optional[str] = "data/taxonomy/mood_food_taxonomy.json",
        max_length: int = 64,
        multi_label: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.multi_label = multi_label

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.normpath(os.path.join(self.script_dir, '../../..'))  # Up to AI_moodfood2
        self.data_path = os.path.join(self.project_root, data_path) if not os.path.isabs(data_path) else data_path
        self.taxonomy_path = (
            taxonomy_path if taxonomy_path
            else os.path.join(self.project_root, "data/taxonomy/mood_food_taxonomy.json")
        )

        # ✅ Load taxonomy label space
        self.label2idx, self.idx2label = self._load_taxonomy(self.taxonomy_path)
        taxonomy_labels = extract_all_sub_labels(taxonomy_path or "data/taxonomy/mood_food_taxonomy.json")

        # ✅ Load and preprocess dataset
        with open(self.data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.samples = []
        skipped = 0

        for item in raw_data:
            text = item.get("text", "").strip()
            raw_labels = item.get("labels") or item.get("intents") or item.get("label") or []
            if isinstance(raw_labels, str):
                raw_labels = [raw_labels]

            # ✅ Normalize labels (via normalize_dataset.py)
            norm_labels = normalize_labels(raw_labels)
            
            # Keep only labels that exist in taxonomy
            matching = [l for l in norm_labels if l in taxonomy_labels]
            if not matching:
                skipped += 1
                continue

            label_ids = (
                [self.label2idx[l] for l in matching]
                if self.multi_label else
                self.label2idx[matching[0]]
            )
            self.samples.append({"text": text, "label_ids": label_ids})

        if skipped:
            print(f"⚠️  {skipped} samples skipped because no labels matched taxonomy "
                  f"(you can map them via normalize_dataset.SYNONYM_MAP).")
            

    def _load_taxonomy(self, taxonomy_path: str):
        print(f"Attempting to load taxonomy from absolute path: {os.path.abspath(taxonomy_path)}")
        with open(taxonomy_path or "AI_moodfood2/data/taxonomy/mood_food_taxonomy.json", "r", encoding="utf-8") as f:
            taxonomy = json.load(f)

        label_set = set()
        for group, moods in taxonomy.items():
            if isinstance(moods, list):
                for mood in moods:
                    label_set.add(mood.strip().lower().replace(" ", "_"))
            elif isinstance(moods, dict):
                descriptors = moods.get("descriptors", [])
                if isinstance(descriptors, str):
                    descriptors = [descriptors]
                for mood in descriptors:
                    label_set.add(mood.strip().lower().replace(" ", "_"))

        label2idx = {label: i for i, label in enumerate(sorted(label_set))}
        idx2label = {i: label for label, i in label2idx.items()}
        print(f"📚 Loaded {len(label2idx)} taxonomy labels.")
        return label2idx, idx2label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        label_tensor = (
            self._get_multi_hot(item["label_ids"])
            if self.multi_label else
            torch.tensor(item["label_ids"], dtype=torch.long)
        )

        return {
            "input_ids": torch.tensor(encoding["input_ids"]).view(-1),
            "attention_mask": torch.tensor(encoding["attention_mask"]).view(-1),
            "labels": label_tensor,
        }

    def _get_multi_hot(self, label_ids: List[int]):
        vec = torch.zeros(len(self.label2idx), dtype=torch.float)
        vec[label_ids] = 1.0
        return vec

    def get_label_encoder(self):
        return self.label2idx, self.idx2label