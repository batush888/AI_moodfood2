# scripts/train_intent_model.py
import json
import os
from typing import Dict, Set, List
import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from transformers import (
    DistilBertTokenizerFast,
    DistilBertModel,
    Trainer,
    TrainingArguments,
)
from transformers.trainer import Trainer
from transformers import TrainingArguments
from transformers.data.data_collator import DataCollatorWithPadding
import logging
from transformers import DistilBertTokenizerFast
from utils.label_utils import DATASET_LABEL_TO_ID, normalize_labels
from core.nlu.utils import validate_intent_dataset


DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased", cache_dir="./models")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# helpers
# -----------------------
def extract_all_sub_labels(taxonomy_path: str) -> Set[str]:
    """
    Robustly extract all sub-labels from taxonomy.
    Handles the formats used in your project and normalizes labels.
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

def normalize_label(s: str) -> str:
    return s.strip().lower().replace(" ", "_")

# -----------------------
# config / paths
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TAXONOMY_PATH = os.path.join(BASE_DIR, "data", "taxonomy", "mood_food_taxonomy.json")
DATA_PATH = os.path.join(BASE_DIR, "../data/intent_dataset.json")
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "../models/intent_classifier")
LOG_DIR = os.path.join(BASE_DIR, "../logs")
DATASET_PATH = os.path.join(BASE_DIR, "../data/intent_dataset.json")

# Validate dataset
validate_intent_dataset(DATASET_PATH, TAXONOMY_PATH)

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------
# Step 1: Load taxonomy and dataset
# -----------------------
with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
    taxonomy = json.load(f)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

logger.info("Loaded raw data size: %d", len(raw_data))
logger.info("Example samples: %s", raw_data[:3])

# -----------------------
# Step 2: build taxonomy-based label space
# -----------------------
taxonomy_labels = sorted(extract_all_sub_labels(TAXONOMY_PATH))
label_to_index = {label: idx for idx, label in enumerate(taxonomy_labels)}
num_taxonomy_labels = len(taxonomy_labels)
logger.info("Taxonomy labels (count=%d) loaded.", num_taxonomy_labels)

# -----------------------
# Step 3: Prepare texts + both label encodings
# -----------------------
texts: List[str] = []
taxonomy_multi_hot_list: List[np.ndarray] = []
mlb_label_lists: List[List[str]] = []

for item in raw_data:
    text = item.get("text", "").strip()
    raw_labels = item.get("labels") or item.get("intents") or item.get("label") or []
    if isinstance(raw_labels, str):
        raw_labels = [raw_labels]

    norm_labels = [normalize_label(l) for l in raw_labels]
    # taxonomy multi-hot (only keep taxonomy-known labels)
    taxonomy_vec = np.zeros(num_taxonomy_labels, dtype=int)
    for l in norm_labels:
        if l in label_to_index:
            taxonomy_vec[label_to_index[l]] = 1

    # skip samples that have no taxonomy match (we rely on taxonomy as primary label)
    if taxonomy_vec.sum() == 0:
        continue

    texts.append(text)
    taxonomy_multi_hot_list.append(taxonomy_vec)
    mlb_label_lists.append(norm_labels)  # keep full normalized labels for MLB

if len(texts) == 0:
    raise RuntimeError("No training samples remained after filtering by taxonomy labels. "
                       "Map/normalize your dataset labels to the taxonomy or update the taxonomy.")

taxonomy_multi_hot = np.stack(taxonomy_multi_hot_list, axis=0)
logger.info("After filtering, samples count=%d", len(texts))
logger.info("Taxonomy multi-hot shape: %s", taxonomy_multi_hot.shape)

# -----------------------
# Step 4: MultiLabelBinarizer for dataset-driven labels (MLB)
# -----------------------
mlb = MultiLabelBinarizer()
mlb_encoded = mlb.fit_transform(mlb_label_lists)  # shape = (n_samples, n_mlb_classes)
num_mlb_labels = mlb_encoded.shape[1]
logger.info("MLB classes: %d", num_mlb_labels)

# Save MLB classes for later inference use
with open(os.path.join(MODEL_OUTPUT_DIR, "mlb_classes.json"), "w", encoding="utf-8") as f:
    json.dump(mlb.classes_.tolist(), f, indent=2)

# -----------------------
# Step 5: train-test split (same split for both label sets)
# -----------------------
train_texts, val_texts, train_taxo, val_taxo, train_mlb, val_mlb = train_test_split(
    texts, taxonomy_multi_hot, mlb_encoded, test_size=0.2, random_state=42
)

logger.info("Train/Val sizes: %d / %d", len(train_texts), len(val_texts))

# -----------------------
# Step 6: Tokenize (Local or Internet)
# -----------------------
def load_tokenizer_and_model_name(base_model_name="distilbert-base-uncased", local_dir="./models/base_model"):
    """
    Load tokenizer from local directory if available, else download from internet and cache locally.
    """
    if os.path.isdir(local_dir) and os.path.exists(os.path.join(local_dir, "tokenizer.json")):
        logger.info(f"📦 Loading tokenizer from local directory: {local_dir}")
        tokenizer = DistilBertTokenizerFast.from_pretrained(local_dir)
        model_source = local_dir
    else:
        logger.info(f"🌐 Downloading tokenizer from Hugging Face model hub: {base_model_name}")
        tokenizer = DistilBertTokenizerFast.from_pretrained(base_model_name, cache_dir=local_dir)
        model_source = base_model_name
    return tokenizer, model_source

tokenizer, model_source = load_tokenizer_and_model_name()

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# -----------------------
# Step 7: Dataset wrapper that returns both labels
# -----------------------
class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, taxonomy_labels, mlb_labels):
        self.encodings = encodings
        self.taxonomy_labels = taxonomy_labels
        self.mlb_labels = mlb_labels

    def __len__(self):
        return len(self.taxonomy_labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.taxonomy_labels[idx], dtype=torch.float)   # primary labels for Trainer
        item["mlb_labels"] = torch.tensor(self.mlb_labels[idx], dtype=torch.float)    # additional labels forwarded to model
        return item

train_dataset = MultiLabelDataset(train_encodings, train_taxo, train_mlb)
val_dataset = MultiLabelDataset(val_encodings, val_taxo, val_mlb)

# -----------------------
# Step 8: Dual-head model (shared encoder + two heads)
# -----------------------
class DistilBertDualHead(nn.Module):
    def __init__(self, model_name_or_path, taxonomy_classes: int, mlb_classes: int, dropout: float = 0.2):
        super().__init__()
        # Load encoder from local cache if available, else from hub
        if os.path.isdir(model_name_or_path) and os.path.exists(os.path.join(model_name_or_path, "config.json")):
            logger.info(f"📦 Loading DistilBERT encoder from local path: {model_name_or_path}")
            self.encoder = DistilBertModel.from_pretrained(model_name_or_path)
        else:
            logger.info(f"🌐 Downloading DistilBERT encoder from Hugging Face: {model_name_or_path}")
            self.encoder = DistilBertModel.from_pretrained(model_name_or_path, cache_dir="./models/base_model")

        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.taxonomy_head = nn.Linear(hidden_size, taxonomy_classes)
        self.mlb_head = nn.Linear(hidden_size, mlb_classes)

    def forward(self, input_ids=None, attention_mask=None, labels=None, mlb_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)

        taxonomy_logits = self.taxonomy_head(pooled)
        mlb_logits = self.mlb_head(pooled)

        loss = None
        if labels is not None:
            bce = BCEWithLogitsLoss()
            taxonomy_loss = bce(taxonomy_logits, labels.float())
            loss = taxonomy_loss
            if mlb_labels is not None:
                mlb_loss = bce(mlb_logits, mlb_labels.float())
                loss = taxonomy_loss + mlb_loss

        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(
            loss=loss,
            logits=taxonomy_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# Instantiate model using either local or downloaded encoder
model = DistilBertDualHead(
    model_name_or_path=model_source,
    taxonomy_classes=num_taxonomy_labels,
    mlb_classes=num_mlb_labels,
)

# -----------------------
# Step 9: training args
# -----------------------
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    eval_strategy="epoch",    # if your HF version complains, try eval_strategy="epoch" or eval_strategy param name used by your transformers
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir=LOG_DIR,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# -----------------------
# Step 10: metrics
# -----------------------
def compute_metrics(eval_pred) -> Dict[str, float]:
    # eval_pred: (predictions, label_ids) where predictions are taxonomy logits
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    labels = labels.astype(int)
    return {
        "f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "precision": float(precision_score(labels, preds, average="micro", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="micro", zero_division=0)),
    }

# -----------------------
# Step 11: Trainer
# -----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# -----------------------
# Step 12: Train + Save artifacts
# -----------------------
trainer.train()

# Save model, tokenizer, and helper artifacts
trainer.save_model(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

# Save taxonomy label mapping and MLB classes
with open(os.path.join(MODEL_OUTPUT_DIR, "taxonomy_labels.json"), "w", encoding="utf-8") as f:
    json.dump(taxonomy_labels, f, indent=2)
with open(os.path.join(MODEL_OUTPUT_DIR, "mlb_classes.json"), "w", encoding="utf-8") as f:
    json.dump(mlb.classes_.tolist(), f, indent=2)

logger.info("✅ Training complete. Model saved to %s", MODEL_OUTPUT_DIR)