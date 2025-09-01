#!/usr/bin/env python3
"""
Simplified model training script compatible with current transformers version.
Focuses on fine-tuning existing model with dish dataset.
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
from sklearn.metrics import classification_report, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertModel,
    DistilBertForSequenceClassification,
    TrainingArguments, 
    Trainer
)
import warnings
warnings.filterwarnings("ignore")

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {DEVICE}")

class SimpleDishDataset(Dataset):
    """Simplified dataset for dish-based training."""
    
    def __init__(self, dish_dataset_path: str, max_length: int = 128):
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        
        # Load dish data
        self.samples = []
        self.labels = []
        self._load_dish_data(dish_dataset_path)
        
        # Setup label encoder
        self._setup_label_encoder()
        
        # Encode labels
        self.encoded_labels = self.mlb.transform(self.labels)
        
        print(f"🎯 Dataset ready: {len(self.samples)} samples, {len(self.mlb.classes_)} labels")

    def _load_dish_data(self, dataset_path: str):
        """Load dish dataset and create training samples."""
        try:
            if dataset_path.endswith('.parquet'):
                df = pd.read_parquet(dataset_path)
            elif dataset_path.endswith('.jsonl'):
                df = pd.read_json(dataset_path, lines=True)
            else:
                raise ValueError("Unsupported file format. Use .parquet or .jsonl")
            
            print(f"📚 Loaded {len(df)} dishes from dataset")
            
            for _, row in df.iterrows():
                dish_name = row['dish_name']
                contexts = row.get('contexts', [])
                ingredients = row.get('ingredients', [])
                
                # Create training samples
                samples_for_dish = [
                    f"I want {dish_name.lower()}",
                    f"Recommend {dish_name.lower()}",
                    f"Something like {dish_name.lower()}",
                    f"Craving {dish_name.lower()}",
                ]
                
                # Add ingredient-based samples
                if ingredients:
                    main_ingredients = ingredients[:2]  # Top 2 ingredients
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
                                f"Something cozy like {dish_name.lower()}",
                            ])
                        elif context == 'EXPERIMENTAL_FLAVOR':
                            samples_for_dish.extend([
                                f"I want something spicy like {dish_name.lower()}",
                                f"Something bold like {dish_name.lower()}",
                            ])
                        elif context == 'ENERGY_VITALITY':
                            samples_for_dish.extend([
                                f"Need energy like {dish_name.lower()}",
                                f"Post-workout meal like {dish_name.lower()}",
                            ])
                
                # Add all samples with their contexts as labels
                for sample in samples_for_dish:
                    self.samples.append(sample)
                    self.labels.append(contexts if contexts else ['GENERAL_RECOMMENDATION'])
                    
        except Exception as e:
            print(f"⚠️ Error loading dish dataset: {e}")

    def _setup_label_encoder(self):
        """Setup multi-label encoder."""
        all_labels = set()
        
        # Collect all unique labels
        for label_list in self.labels:
            if isinstance(label_list, list):
                all_labels.update(label_list)
            else:
                all_labels.add(label_list)
        
        # Ensure we have default labels
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
        for i, label in enumerate(self.mlb.classes_):
            print(f"   {i+1:2d}. {label}")

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


def simple_train_model(
    dish_dataset_path: str,
    output_dir: str = "models/simple_intent_classifier",
    num_epochs: int = 4,
    batch_size: int = 8,
    learning_rate: float = 3e-5
):
    """Simple model training with basic configuration."""
    
    print("🚀 Starting simple model training...")
    
    # Create dataset
    dataset = SimpleDishDataset(dish_dataset_path)
    
    if len(dataset) == 0:
        print("❌ No training samples found.")
        return
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Initialize model using standard DistilBERT
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(dataset.mlb.classes_),
        problem_type="multi_label_classification"
    )
    
    # Move model to device
    model.to(DEVICE)
    
    # Simple training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
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
    
    # Save training info
    training_info = {
        'num_labels': len(dataset.mlb.classes_),
        'classes': dataset.mlb.classes_.tolist(),
        'total_samples': len(dataset),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
    }
    
    with open(f"{output_dir}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print("✅ Training completed successfully!")
    print(f"📁 Model saved to: {output_dir}")
    print(f"🎯 Total labels: {len(dataset.mlb.classes_)}")
    
    return trainer, model, dataset


def test_model(model_dir: str):
    """Test the trained model with sample queries."""
    print(f"🧪 Testing model from: {model_dir}")
    
    # Load components
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    
    with open(f"{model_dir}/label_encoder.pkl", "rb") as f:
        mlb = pickle.load(f)
    
    with open(f"{model_dir}/training_info.json", "r") as f:
        info = json.load(f)
    
    # Load model
    model = DistilBertForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=info['num_labels'],
        problem_type="multi_label_classification"
    )
    model.to(DEVICE)
    model.eval()
    
    # Test queries
    test_queries = [
        "I want something spicy and bold",
        "Need comfort food for a cozy night", 
        "Post-workout meal with protein",
        "Light and healthy meal",
        "Something for a hot summer day",
        "Romantic dinner for two",
        "I want Mapo Tofu",
        "Craving Braised Pork Belly",
        "Something with garlic and ginger",
        "Need energy boost",
    ]
    
    print("\n🧪 Model Predictions:")
    print("-" * 80)
    
    with torch.no_grad():
        for query in test_queries:
            # Tokenize
            inputs = tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            
            # Move to device
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # Predict
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Apply sigmoid and threshold
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).cpu().numpy()[0]
            
            # Get predicted labels
            predicted_labels = [mlb.classes_[i] for i, pred in enumerate(predictions) if pred]
            
            # Get top 3 predictions with confidence
            confidence_scores = probs.cpu().numpy()[0]
            top_predictions = sorted(
                [(mlb.classes_[i], score) for i, score in enumerate(confidence_scores)],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            print(f"Query: {query}")
            print(f"Predicted: {predicted_labels}")
            print(f"Top 3: {top_predictions}")
            print("-" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple model training with dish dataset")
    parser.add_argument("--dish-dataset", type=str, default="data/processed/dish_dataset.parquet", 
                       help="Path to dish dataset")
    parser.add_argument("--output-dir", type=str, default="models/simple_intent_classifier",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--test-only", action="store_true", help="Only test existing model")
    
    args = parser.parse_args()
    
    if args.test_only:
        test_model(args.output_dir)
    else:
        trainer, model, dataset = simple_train_model(
            dish_dataset_path=args.dish_dataset,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # Test the model
        print("\n" + "="*80)
        test_model(args.output_dir)
