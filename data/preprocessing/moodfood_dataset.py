import json
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import DistilBertTokenizerFast
import torch

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

LABEL_TO_TAXONOMY = {
    "temperature_warm": "COMFORT_EMOTIONAL",
    "goal_comfort": "COMFORT_EMOTIONAL",
    "emotion_sad": "COMFORT_EMOTIONAL",
    "sensory_soothing": "COMFORT_EMOTIONAL",
    "goal_energy": "ENERGY_VITALITY",
    "emotion_adventurous": "ENERGY_VITALITY",
    "sensory_exotic": "ENERGY_VITALITY",
    "goal_light": "HEALTH_CLEAN",
    "goal_healthy": "HEALTH_CLEAN",
    "temperature_cold": "HEALTH_CLEAN",
    "food_smoothie": "ENERGY_VITALITY",
    "occasion_romantic": "ROMANTIC_DINNER",
    "meal_dinner": "ROMANTIC_DINNER",
    "social_group": "ROMANTIC_DINNER"
}

class MoodFoodDataset(torch.utils.data.Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
        
        # Map labels to taxonomy categories
        self.texts = []
        self.labels = []
        for entry in raw_data:
            mapped_labels = set()
            for label in entry["labels"]:
                if label in LABEL_TO_TAXONOMY:
                    mapped_labels.add(LABEL_TO_TAXONOMY[label])
            if mapped_labels:
                self.texts.append(entry["text"])
                self.labels.append(list(mapped_labels))

        # Fit multi-label binarizer on all taxonomy categories (keys)
        all_taxonomy_labels = list(set(LABEL_TO_TAXONOMY.values()))
        self.mlb = MultiLabelBinarizer(classes=all_taxonomy_labels)
        self.labels_bin = self.mlb.fit_transform(self.labels)
        if hasattr(self.labels_bin, "toarray"):
            self.labels_bin = self.labels_bin.toarray()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=64, return_tensors='pt')
        label = torch.tensor(self.labels_bin[idx].toarray().flatten() if hasattr(self.labels_bin[idx], "toarray") else self.labels_bin[idx], dtype=torch.float)

        # Proper tensor shape adjustment
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label
        }