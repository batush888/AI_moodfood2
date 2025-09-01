#!/usr/bin/env python3
"""
Enhanced training script for intent classification using dish dataset.
Combines the original intent dataset with the new dish dataset for improved performance.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertModel,
    DistilBertForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EvalPrediction
)
import warnings
warnings.filterwarnings("ignore")

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {DEVICE}")

class EnhancedMoodFoodDataset(Dataset):
    """Enhanced dataset combining original intent data with dish data."""
    
    def __init__(self, 
                 intent_dataset_path: str = None,
                 dish_dataset_path: str = None,
                 taxonomy_path: str = None,
                 max_length: int = 128):
        
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        
        # Load data sources
        self.samples = []
        self.labels = []
        
        # Load original intent dataset if available
        if intent_dataset_path and os.path.exists(intent_dataset_path):
            self._load_intent_dataset(intent_dataset_path)
            print(f"📚 Loaded {len(self.samples)} samples from intent dataset")
        
        # Load dish dataset
        if dish_dataset_path and os.path.exists(dish_dataset_path):
            self._load_dish_dataset(dish_dataset_path)
            print(f"📚 Total samples after adding dish data: {len(self.samples)}")
        
        # Setup label encoder
        self._setup_label_encoder(taxonomy_path)
        
        # Encode labels
        self.encoded_labels = self.mlb.transform(self.labels)
        
        print(f"🎯 Dataset ready: {len(self.samples)} samples, {len(self.mlb.classes_)} labels")

    def _load_intent_dataset(self, dataset_path: str):
        """Load original intent classification dataset."""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                query = item.get('query', '').strip()
                if query:
                    # Convert single intent to list for consistency
                    intent = item.get('intent', item.get('label', 'unknown'))
                    if isinstance(intent, str):
                        intent = [intent]
                    
                    self.samples.append(query)
                    self.labels.append(intent)
                    
        except Exception as e:
            print(f"⚠️ Error loading intent dataset: {e}")

    def _load_dish_dataset(self, dataset_path: str):
        """Load dish dataset from JSONL or Parquet."""
        try:
            if dataset_path.endswith('.parquet'):
                df = pd.read_parquet(dataset_path)
            elif dataset_path.endswith('.jsonl'):
                df = pd.read_json(dataset_path, lines=True)
            else:
                raise ValueError("Unsupported file format. Use .parquet or .jsonl")
            
            for _, row in df.iterrows():
                # Create training samples from dish data
                dish_name = row['dish_name']
                contexts = row.get('contexts', [])
                ingredients = row.get('ingredients', [])
                nutrients = row.get('nutrients', {})
                
                # Generate diverse training samples
                samples_for_dish = [
                    f"I want {dish_name.lower()}",
                    f"Recommend {dish_name.lower()}",
                    f"Something like {dish_name.lower()}",
                    f"Craving {dish_name.lower()}",
                ]
                
                # Add ingredient-based samples
                if ingredients:
                    main_ingredients = ingredients[:3]  # Top 3 ingredients
                    samples_for_dish.extend([
                        f"Something with {', '.join(main_ingredients)}",
                        f"Dish containing {main_ingredients[0]}",
                    ])
                
                # Add context-based samples
                if contexts:
                    for context in contexts:
                        if context == 'COMFORT_EMOTIONAL':
                            samples_for_dish.extend([
                                f"I need comfort food like {dish_name.lower()}",
                                f"Something cozy and warm like {dish_name.lower()}",
                            ])
                        elif context == 'EXPERIMENTAL_FLAVOR':
                            samples_for_dish.extend([
                                f"I want something spicy like {dish_name.lower()}",
                                f"Something bold and adventurous like {dish_name.lower()}",
                            ])
                        elif context == 'ENERGY_VITALITY':
                            samples_for_dish.extend([
                                f"Need energy boost like {dish_name.lower()}",
                                f"Post-workout meal like {dish_name.lower()}",
                            ])
                
                # Add all samples with their contexts as labels
                for sample in samples_for_dish:
                    self.samples.append(sample)
                    self.labels.append(contexts if contexts else ['GENERAL_RECOMMENDATION'])
                    
        except Exception as e:
            print(f"⚠️ Error loading dish dataset: {e}")

    def _setup_label_encoder(self, taxonomy_path: str = None):
        """Setup multi-label encoder based on taxonomy and data."""
        all_labels = set()
        
        # Collect all unique labels
        for label_list in self.labels:
            if isinstance(label_list, list):
                all_labels.update(label_list)
            else:
                all_labels.add(label_list)
        
        # Load taxonomy labels if available
        if taxonomy_path and os.path.exists(taxonomy_path):
            try:
                with open(taxonomy_path, 'r', encoding='utf-8') as f:
                    taxonomy = json.load(f)
                
                # Extract all taxonomy labels
                taxonomy_labels = set()
                for category in taxonomy.values():
                    if isinstance(category, dict) and 'subcategories' in category:
                        for subcat in category['subcategories'].values():
                            if isinstance(subcat, dict) and 'examples' in subcat:
                                taxonomy_labels.add(subcat.get('id', '').upper())
                
                all_labels.update(taxonomy_labels)
                print(f"📋 Added {len(taxonomy_labels)} labels from taxonomy")
                
            except Exception as e:
                print(f"⚠️ Error loading taxonomy: {e}")
        
        # Ensure we have some default labels
        default_labels = {
            'COMFORT_EMOTIONAL', 'ENERGY_VITALITY', 'EXPERIMENTAL_FLAVOR',
            'HEALTH_LIGHT', 'WEATHER_HOT', 'WEATHER_COLD', 'SOCIAL_ROMANTIC',
            'SOCIAL_GROUP', 'SOCIAL_FAMILY', 'GENERAL_RECOMMENDATION'
        }
        all_labels.update(default_labels)
        
        # Sort labels for consistency
        sorted_labels = sorted(list(all_labels))
        
        # Initialize MultiLabelBinarizer
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([sorted_labels])
        
        print(f"🏷️ Label encoder ready with {len(self.mlb.classes_)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.encoded_labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            sample,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }


class DistilBertDualHead(nn.Module):
    """Enhanced DistilBERT with dual classification heads."""
    
    def __init__(self, model_name_or_path, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.num_labels = num_labels
        self.encoder = DistilBertModel.from_pretrained(model_name_or_path)
        hidden_size = self.encoder.config.hidden_size
        
        # Enhanced architecture with better regularization
        self.dropout = nn.Dropout(dropout)
        self.intermediate = nn.Linear(hidden_size, hidden_size // 2)
        self.activation = nn.GELU()
        self.classifier = nn.Linear(hidden_size // 2, num_labels)
        self.layer_norm = nn.LayerNorm(hidden_size // 2)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # Enhanced forward pass
        pooled_output = self.dropout(pooled_output)
        intermediate = self.intermediate(pooled_output)
        intermediate = self.layer_norm(intermediate)
        intermediate = self.activation(intermediate)
        intermediate = self.dropout(intermediate)
        
        logits = self.classifier(intermediate)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        
        return {
            'loss': loss,
            'logits': logits,
        }


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute metrics for multi-label classification."""
    predictions, labels = eval_pred
    
    # Apply sigmoid to get probabilities
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    
    # Apply threshold to get binary predictions
    threshold = 0.5
    y_pred = (probs > threshold).numpy()
    y_true = labels
    
    # Calculate metrics
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall,
    }


def train_enhanced_model(
    intent_dataset_path: str = None,
    dish_dataset_path: str = None,
    taxonomy_path: str = None,
    output_dir: str = "models/enhanced_intent_classifier",
    num_epochs: int = 8,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1
):
    """Train enhanced intent classification model."""
    
    print("🚀 Starting enhanced model training...")
    
    # Create dataset
    dataset = EnhancedMoodFoodDataset(
        intent_dataset_path=intent_dataset_path,
        dish_dataset_path=dish_dataset_path,
        taxonomy_path=taxonomy_path
    )
    
    if len(dataset) == 0:
        print("❌ No training samples found. Check your dataset paths.")
        return
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"📊 Training samples: {train_size}")
    print(f"📊 Validation samples: {val_size}")
    print(f"🎯 Number of labels: {len(dataset.mlb.classes_)}")
    
    # Initialize model
    model = DistilBertDualHead(
        model_name_or_path="distilbert-base-uncased",
        num_labels=len(dataset.mlb.classes_)
    )
    
    # Move model to device
    model.to(DEVICE)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        evaluation_strategy="epoch",  # Fixed: use evaluation_strategy instead of eval_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
        dataloader_pin_memory=False,
        gradient_accumulation_steps=2,
        fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA available
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train model
    print("🏋️ Starting training...")
    trainer.train()
    
    # Save model and components
    print("💾 Saving model and components...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model.save_pretrained(output_dir)
    
    # Save tokenizer
    dataset.tokenizer.save_pretrained(output_dir)
    
    # Save label encoder
    with open(f"{output_dir}/label_encoder.pkl", "wb") as f:
        pickle.dump(dataset.mlb, f)
    
    # Save label mapping for reference
    label_mapping = {
        'classes': dataset.mlb.classes_.tolist(),
        'num_classes': len(dataset.mlb.classes_),
        'training_samples': len(dataset),
    }
    
    with open(f"{output_dir}/label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=2)
    
    # Save training summary
    training_summary = {
        'model_type': 'DistilBertDualHead',
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'total_samples': len(dataset),
        'train_samples': train_size,
        'val_samples': val_size,
        'num_labels': len(dataset.mlb.classes_),
        'intent_dataset': intent_dataset_path,
        'dish_dataset': dish_dataset_path,
        'taxonomy_path': taxonomy_path,
        'device': str(DEVICE),
    }
    
    with open(f"{output_dir}/training_summary.json", "w") as f:
        json.dump(training_summary, f, indent=2)
    
    print("✅ Training completed successfully!")
    print(f"📁 Model saved to: {output_dir}")
    print(f"🎯 Total labels: {len(dataset.mlb.classes_)}")
    print(f"📊 Training samples: {len(dataset)}")
    
    return trainer, model, dataset


def evaluate_model(model_dir: str, test_samples: List[Tuple[str, List[str]]] = None):
    """Evaluate the trained model."""
    print(f"🧪 Evaluating model from: {model_dir}")
    
    # Load model components
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    
    with open(f"{model_dir}/label_encoder.pkl", "rb") as f:
        mlb = pickle.load(f)
    
    # Load model architecture info
    with open(f"{model_dir}/label_mapping.json", "r") as f:
        label_info = json.load(f)
    
    # Initialize model
    model = DistilBertDualHead(
        model_name_or_path="distilbert-base-uncased",
        num_labels=label_info['num_classes']
    )
    
    # Load trained weights
    try:
        # Try loading safetensors format first (newer transformers)
        from safetensors.torch import load_file
        model.load_state_dict(load_file(f"{model_dir}/model.safetensors"))
    except:
        # Fallback to pytorch_model.bin
        model.load_state_dict(torch.load(f"{model_dir}/pytorch_model.bin", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # Test samples
    if test_samples is None:
        test_samples = [
            ("I want something spicy and bold", ["EXPERIMENTAL_FLAVOR"]),
            ("Need comfort food for a cozy night", ["COMFORT_EMOTIONAL"]),
            ("Post-workout meal with protein", ["ENERGY_VITALITY"]),
            ("Light and healthy meal", ["HEALTH_LIGHT"]),
            ("Something for a hot summer day", ["WEATHER_HOT"]),
            ("Romantic dinner for two", ["SOCIAL_ROMANTIC"]),
            ("I want Mapo Tofu", ["EXPERIMENTAL_FLAVOR", "ENERGY_VITALITY"]),
            ("Craving Braised Pork Belly", ["COMFORT_EMOTIONAL"]),
        ]
    
    print("\n🧪 Testing model predictions:")
    print("-" * 80)
    
    with torch.no_grad():
        for text, expected_labels in test_samples:
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            
            # Move to device
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # Predict
            outputs = model(**inputs)
            logits = outputs['logits']
            
            # Apply sigmoid and threshold
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).cpu().numpy()[0]
            
            # Get predicted labels
            predicted_labels = [mlb.classes_[i] for i, pred in enumerate(predictions) if pred]
            
            # Calculate confidence scores
            confidence_scores = probs.cpu().numpy()[0]
            top_predictions = sorted(
                [(mlb.classes_[i], score) for i, score in enumerate(confidence_scores)],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            print(f"Input: {text}")
            print(f"Expected: {expected_labels}")
            print(f"Predicted: {predicted_labels}")
            print(f"Top predictions: {top_predictions}")
            print("-" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train enhanced intent classification model")
    parser.add_argument("--intent-dataset", type=str, help="Path to original intent dataset")
    parser.add_argument("--dish-dataset", type=str, default="data/processed/dish_dataset.parquet", 
                       help="Path to dish dataset (parquet or jsonl)")
    parser.add_argument("--taxonomy", type=str, default="data/taxonomy/mood_food_taxonomy.json",
                       help="Path to taxonomy file")
    parser.add_argument("--output-dir", type=str, default="models/enhanced_intent_classifier",
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained model")
    
    args = parser.parse_args()
    
    if args.evaluate:
        evaluate_model(args.output_dir)
    else:
        trainer, model, dataset = train_enhanced_model(
            intent_dataset_path=args.intent_dataset,
            dish_dataset_path=args.dish_dataset,
            taxonomy_path=args.taxonomy,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # Run evaluation after training
        print("\n" + "="*80)
        evaluate_model(args.output_dir)
