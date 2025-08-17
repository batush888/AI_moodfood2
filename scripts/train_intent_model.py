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
from sklearn.utils.class_weight import compute_class_weight

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
from utils.label_utils import (
    load_taxonomy,
    load_dataset_labels,
    get_label_mappings_from_taxonomy_and_dataset,
    get_reverse_label_mappings,
    normalize_labels
)
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
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # Go up one level to project root
TAXONOMY_PATH = os.path.join(PROJECT_ROOT, "data", "taxonomy", "mood_food_taxonomy.json")
DATA_PATH = os.path.join(PROJECT_ROOT, "data/intent_dataset.json")
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models/intent_classifier")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
DATASET_PATH = os.path.join(PROJECT_ROOT, "data/intent_dataset.json")

# Validate dataset - now enabled with auto-normalized taxonomy
validate_intent_dataset(DATASET_PATH, TAXONOMY_PATH)

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------
# Step 1: Load taxonomy and dataset using reusable functions
# -----------------------
taxonomy = load_taxonomy()
dataset_labels = load_dataset_labels()

# Load raw dataset data
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

logger.info("Loaded taxonomy with %d categories", len(taxonomy))
logger.info("Loaded %d unique dataset labels", len(dataset_labels))
logger.info("Loaded raw data size: %d", len(raw_data))
logger.info("Example samples: %s", raw_data[:3])

# -----------------------
# Step 2: build unified label space using reusable functions
# -----------------------
# Create unified mapping from taxonomy and dataset
unified_label_to_id = get_label_mappings_from_taxonomy_and_dataset(taxonomy, dataset_labels)
unified_id_to_label = get_reverse_label_mappings(unified_label_to_id)

# For taxonomy-specific training, we'll use the taxonomy keys
taxonomy_labels = sorted(taxonomy.keys())
label_to_index = {label: idx for idx, label in enumerate(taxonomy_labels)}
num_taxonomy_labels = len(taxonomy_labels)

logger.info("Unified labels (count=%d) created.", len(unified_label_to_id))
logger.info("Taxonomy labels (count=%d) loaded.", num_taxonomy_labels)
logger.info("Dataset labels (count=%d) loaded.", len(dataset_labels))

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

    # Normalize labels to match taxonomy format
    norm_labels = [label.lower().replace(" ", "_") for label in raw_labels]
    # taxonomy multi-hot (only keep taxonomy-known labels)
    taxonomy_vec = np.zeros(num_taxonomy_labels, dtype=int)
    for l in norm_labels:
        if l in label_to_index:
            taxonomy_vec[label_to_index[l]] = 1

    # Keep samples that have at least one taxonomy match
    # This allows training with the limited common labels we have
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
# Step 7.5: Compute enhanced class weights for balanced training
# -----------------------
# Convert to numpy for easier manipulation
train_taxo_np = np.array(train_taxo)

# Compute per-label positive and negative counts
pos_counts = torch.tensor(train_taxo_np.sum(axis=0), dtype=torch.float)  # [C]
neg_counts = torch.tensor(train_taxo_np.shape[0] - train_taxo_np.sum(axis=0), dtype=torch.float)

# Avoid div-by-zero; cap min positives at 1
pos_counts = torch.clamp(pos_counts, min=1.0)
pos_weight = (neg_counts / pos_counts)  # [C]

# Log the computed weights for debugging
logger.info(f"Computed per-label class weights:")
for i, (pos, neg, weight) in enumerate(zip(pos_counts, neg_counts, pos_weight)):
    if pos > 1:  # Only log active classes
        logger.info(f"  Label {i}: pos={pos:.0f}, neg={neg:.0f}, weight={weight:.3f}")

class_weights_tensor = pos_weight
logger.info(f"Class weights range: {pos_weight.min():.3f} - {pos_weight.max():.3f}")

# -----------------------
# Step 8: Dual-head model (shared encoder + two heads)
# -----------------------
class DistilBertDualHead(nn.Module):
    def __init__(self, model_name_or_path, taxonomy_classes: int, mlb_classes: int, class_weights: torch.Tensor, dropout: float = 0.2):
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
        self.class_weights = class_weights  # Store class weights for loss computation

    def forward(self, input_ids=None, attention_mask=None, labels=None, mlb_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)

        taxonomy_logits = self.taxonomy_head(pooled)
        mlb_logits = self.mlb_head(pooled)

        loss = None
        if labels is not None:
            # Use weighted BCE loss for taxonomy labels to handle class imbalance
            # Move class weights to the same device as the inputs
            device = taxonomy_logits.device
            class_weights = self.class_weights.to(device)
            taxonomy_bce = BCEWithLogitsLoss(pos_weight=class_weights)
            taxonomy_loss = taxonomy_bce(taxonomy_logits, labels.float())
            loss = taxonomy_loss
            
            if mlb_labels is not None:
                # Use regular BCE loss for MLB labels (no class weights needed)
                mlb_bce = BCEWithLogitsLoss()
                mlb_loss = mlb_bce(mlb_logits, mlb_labels.float())
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
    class_weights=class_weights_tensor,
)

# Note: Class weights will be moved to the correct device when the model is moved by the trainer
logger.info(f"Class weights tensor shape: {class_weights_tensor.shape}")
logger.info(f"Class weights will be moved to model device during training")

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
    
    # Ensure predictions are the right shape
    if isinstance(logits, tuple):
        # Sometimes logits come as a tuple, take the first element
        logits = logits[0]
    
    # Convert logits to predictions
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    
    # Handle different label formats and ensure consistent shape
    try:
        if hasattr(labels, 'astype'):
            labels = labels.astype(int)
        else:
            labels = np.array(labels, dtype=int)
        
        # Ensure labels are 2D for sklearn metrics
        if labels.ndim > 2:
            labels = labels.reshape(labels.shape[0], -1)
        
        # Ensure predictions match labels shape
        if preds.shape != labels.shape:
            logger.warning(f"Shape mismatch: preds {preds.shape} vs labels {labels.shape}")
            # Try to reshape predictions to match labels
            if preds.ndim == 1 and labels.ndim == 2:
                preds = preds.reshape(-1, 1)
            elif preds.ndim == 2 and labels.ndim == 1:
                labels = labels.reshape(-1, 1)
        
        return {
            "f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
            "precision": float(precision_score(labels, preds, average="micro", zero_division=0)),
            "recall": float(recall_score(labels, preds, average="micro", zero_division=0)),
        }
    except Exception as e:
        # Fallback metrics if computation fails
        logger.warning(f"Metrics computation failed: {e}. Using fallback metrics.")
        return {
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "error": str(e)
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

# Save unified label mappings for future use
with open(os.path.join(MODEL_OUTPUT_DIR, "unified_label_mappings.json"), "w", encoding="utf-8") as f:
    json.dump({
        "unified_label_to_id": unified_label_to_id,
        "unified_id_to_label": {str(k): v for k, v in unified_id_to_label.items()},
        "dataset_labels": dataset_labels,
        "taxonomy_categories": list(taxonomy.keys())
    }, f, indent=2)

logger.info("✅ Training complete. Model saved to %s", MODEL_OUTPUT_DIR)

# -----------------------
# Step 13: Per-label threshold tuning for better inference
# -----------------------
logger.info("🎯 Tuning per-label thresholds for optimal F1...")

def tune_per_label_thresholds(val_logits, val_true, grid=None):
    """Tune thresholds per label to maximize F1 score."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)  # coarse → fast; widen if you want
    
    probs = 1.0 / (1.0 + np.exp(-val_logits))
    C = probs.shape[1]
    thresholds = []
    
    for c in range(C):
        best_f1, best_t = -1.0, 0.5
        y_true = val_true[:, c]
        
        for t in grid:
            y_pred = (probs[:, c] >= t).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            if f1 > best_f1:
                best_f1, best_t = f1, t
        
        thresholds.append(best_t)
        logger.info(f"  Label {c}: best threshold={best_t:.3f}, F1={best_f1:.3f}")
    
    return thresholds

try:
    # Get validation predictions for threshold tuning
    val_predictions = trainer.predict(val_dataset)
    
    # Handle different prediction formats
    if hasattr(val_predictions, 'predictions'):
        val_logits = val_predictions.predictions
    elif isinstance(val_predictions, tuple):
        val_logits = val_predictions[0]
    else:
        val_logits = val_predictions
    
    if hasattr(val_predictions, 'label_ids'):
        val_true = val_predictions.label_ids
    elif isinstance(val_predictions, tuple) and len(val_predictions) > 1:
        val_true = val_predictions[1]
    else:
        # Fallback: use validation dataset labels
        val_true = np.array([sample['labels'] for sample in val_dataset])
    
    # Ensure we have the right shapes and convert to numpy if needed
    if hasattr(val_logits, 'ndim'):
        if val_logits.ndim > 2:
            val_logits = val_logits.reshape(val_logits.shape[0], -1)
    else:
        val_logits = np.array(val_logits)
        
    if hasattr(val_true, 'ndim'):
        if val_true.ndim > 2:
            val_true = val_true.reshape(val_true.shape[0], -1)
    else:
        val_true = np.array(val_true)
    
    logger.info(f"Validation data shapes: logits {val_logits.shape}, labels {val_true.shape}")
    
    # Tune thresholds
    per_label_thresholds = tune_per_label_thresholds(val_logits, val_true)
    
    # Save thresholds
    with open(os.path.join(MODEL_OUTPUT_DIR, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(per_label_thresholds, f, indent=2)
    
    logger.info(f"✅ Saved {len(per_label_thresholds)} per-label thresholds")
    
except Exception as e:
    logger.warning(f"Threshold tuning failed: {e}. Using default 0.5 for all labels.")
    # Create default thresholds
    per_label_thresholds = [0.5] * num_taxonomy_labels
    with open(os.path.join(MODEL_OUTPUT_DIR, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(per_label_thresholds, f, indent=2)

# Save class weights for inference
with open(os.path.join(MODEL_OUTPUT_DIR, "pos_weight.json"), "w", encoding="utf-8") as f:
    json.dump(class_weights_tensor.tolist(), f, indent=2)