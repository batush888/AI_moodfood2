import os
import torch
import numpy as np
import json
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from sklearn.metrics import f1_score, precision_score, recall_score
from moodfood_dataset import MoodFoodDataset
from torch.utils.data import random_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Use current working directory instead of relative path
PROJECT_ROOT = os.getcwd()

# -----------------------------
# Load taxonomy labels
# -----------------------------
def extract_all_sub_labels(taxonomy_path: str) -> set[str]:
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    
    all_labels = set()
    for sub_labels in taxonomy.values():
        if isinstance(sub_labels, list):
            all_labels.update(sub_labels)
    return all_labels

# -----------------------------
# Multi-label evaluation metrics
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    return {
        "f1": f1_score(labels, preds, average="micro", zero_division=0),
        "precision": precision_score(labels, preds, average="micro", zero_division=0),
        "recall": recall_score(labels, preds, average="micro", zero_division=0),
    }

# -----------------------------
# Train the intent classifier
# -----------------------------
def train_model():
    # Optional sanity check
    taxonomy_path = os.path.join(PROJECT_ROOT, "data", "taxonomy", "mood_food_taxonomy.json")
    dataset_path = os.path.join(PROJECT_ROOT, "data", "intent_dataset.json")
    
    print(f"Attempting to load dataset from absolute path: {os.path.abspath(dataset_path)}")
    print(f"Attempting to load taxonomy from absolute path: {os.path.abspath(taxonomy_path)}")
    labels_from_taxonomy = extract_all_sub_labels(taxonomy_path)
    print(f"📚 Loaded {len(labels_from_taxonomy)} taxonomy intent classes.")
    valid_labels = labels_from_taxonomy
    print(f"📚 Extracted {len(valid_labels)} sub-labels from taxonomy.")

    # 1. Load dataset
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    dataset = MoodFoodDataset(dataset_path)

    # Check if dataset has samples
    if len(dataset) == 0:
        print("❌ No training samples found. Please check your dataset file.")
        return

    # Create PyTorch datasets
    full_size = len(dataset)
    train_size = int(0.8 * full_size)
    val_size = full_size - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    # Get labels from the MultiLabelBinarizer
    num_labels = len(dataset.mlb.classes_)
    print(f"🎯 Training model with {num_labels} taxonomy categories:")
    for i, label in enumerate(dataset.mlb.classes_):
        print(f"   {i+1:2d}. {label}")

    # 2. Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        problem_type="multi_label_classification",  # Important for HF >=4.49
    )

    # 3. Training configuration
    training_args = TrainingArguments(
        output_dir="./models/intent_model",
        num_train_epochs=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=10,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=1,
        greater_is_better=True,
        report_to="none",
    )

    # 4. Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # 5. Start training
    trainer.train()

    # 6. Save final model, tokenizer, and label encoder
    os.makedirs("./models", exist_ok=True)
    model.save_pretrained("./models/intent_model")
    tokenizer.save_pretrained("./models/intent_model")
    
    # Save the MultiLabelBinarizer for inference
    import pickle
    with open("./models/label_encoder.pkl", "wb") as f:
        pickle.dump(dataset.mlb, f)
    
    print("✅ Model, tokenizer, and label encoder saved successfully!")
    print(f"📁 Model saved to: ./models/intent_model")
    print(f"📁 Label encoder saved to: ./models/label_encoder.pkl")
    print(f"🎯 Total taxonomy categories: {len(dataset.mlb.classes_)}")
    print(f"📊 Training samples: {len(dataset)}")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    train_model()
