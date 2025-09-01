al #!/usr/bin/env python3
"""
Improved model training script with better data preprocessing and longer training.
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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification
)
import warnings
warnings.filterwarnings("ignore")

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {DEVICE}")

class ImprovedDishDataset(Dataset):
    """Improved dataset with better data preprocessing."""
    
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
        """Load dish dataset and create training samples with better preprocessing."""
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
                
                # Clean contexts - remove any malformed entries
                clean_contexts = []
                for ctx in contexts:
                    if isinstance(ctx, str) and ctx.strip() and len(ctx.strip()) > 1:
                        # Only keep valid context labels (not single characters or malformed)
                        if ctx.strip() in ['COMFORT_EMOTIONAL', 'ENERGY_VITALITY', 'EXPERIMENTAL_FLAVOR', 
                                         'HEALTH_LIGHT', 'WEATHER_HOT', 'WEATHER_COLD', 'SOCIAL_ROMANTIC',
                                         'SOCIAL_GROUP', 'SOCIAL_FAMILY', 'GENERAL_RECOMMENDATION']:
                            clean_contexts.append(ctx.strip())
                
                # If no valid contexts, use general recommendation
                if not clean_contexts:
                    clean_contexts = ['GENERAL_RECOMMENDATION']
                
                # Create more diverse training samples
                samples_for_dish = [
                    f"I want {dish_name.lower()}",
                    f"Recommend {dish_name.lower()}",
                    f"Something like {dish_name.lower()}",
                    f"Craving {dish_name.lower()}",
                    f"Give me {dish_name.lower()}",
                    f"I need {dish_name.lower()}",
                    f"Show me {dish_name.lower()}",
                    f"Find {dish_name.lower()}",
                ]
                
                # Add ingredient-based samples
                if ingredients:
                    main_ingredients = ingredients[:3]  # Top 3 ingredients
                    samples_for_dish.extend([
                        f"Something with {', '.join(main_ingredients)}",
                        f"Dish containing {main_ingredients[0]}",
                        f"Food with {main_ingredients[0]} and {main_ingredients[1] if len(main_ingredients) > 1 else 'spices'}",
                        f"I want food with {main_ingredients[0]}",
                    ])
                
                # Add context-based samples with more variety
                for context in clean_contexts:
                    if context == 'COMFORT_EMOTIONAL':
                        samples_for_dish.extend([
                            f"I need comfort food like {dish_name.lower()}",
                            f"Something cozy like {dish_name.lower()}",
                            f"Warm and comforting like {dish_name.lower()}",
                            f"Home-style food like {dish_name.lower()}",
                        ])
                    elif context == 'EXPERIMENTAL_FLAVOR':
                        samples_for_dish.extend([
                            f"I want something spicy like {dish_name.lower()}",
                            f"Something bold like {dish_name.lower()}",
                            f"Adventurous food like {dish_name.lower()}",
                            f"Spicy and hot like {dish_name.lower()}",
                        ])
                    elif context == 'ENERGY_VITALITY':
                        samples_for_dish.extend([
                            f"Need energy like {dish_name.lower()}",
                            f"Post-workout meal like {dish_name.lower()}",
                            f"Protein-rich like {dish_name.lower()}",
                            f"Energizing food like {dish_name.lower()}",
                        ])
                    elif context == 'HEALTH_LIGHT':
                        samples_for_dish.extend([
                            f"Healthy food like {dish_name.lower()}",
                            f"Light meal like {dish_name.lower()}",
                            f"Low-calorie like {dish_name.lower()}",
                            f"Nutritious like {dish_name.lower()}",
                        ])
                    elif context == 'WEATHER_HOT':
                        samples_for_dish.extend([
                            f"Hot weather food like {dish_name.lower()}",
                            f"Summer meal like {dish_name.lower()}",
                            f"Cooling food like {dish_name.lower()}",
                        ])
                    elif context == 'WEATHER_COLD':
                        samples_for_dish.extend([
                            f"Cold weather food like {dish_name.lower()}",
                            f"Winter meal like {dish_name.lower()}",
                            f"Warming food like {dish_name.lower()}",
                        ])
                    elif context == 'SOCIAL_ROMANTIC':
                        samples_for_dish.extend([
                            f"Romantic dinner like {dish_name.lower()}",
                            f"Date night food like {dish_name.lower()}",
                            f"Intimate meal like {dish_name.lower()}",
                        ])
                    elif context == 'SOCIAL_GROUP':
                        samples_for_dish.extend([
                            f"Group meal like {dish_name.lower()}",
                            f"Party food like {dish_name.lower()}",
                            f"Sharing dish like {dish_name.lower()}",
                        ])
                    elif context == 'SOCIAL_FAMILY':
                        samples_for_dish.extend([
                            f"Family meal like {dish_name.lower()}",
                            f"Kid-friendly like {dish_name.lower()}",
                            f"Family dinner like {dish_name.lower()}",
                        ])
                
                # Add all samples with their contexts as labels
                for sample in samples_for_dish:
                    self.samples.append(sample)
                    self.labels.append(clean_contexts)
                    
        except Exception as e:
            print(f"⚠️ Error loading dish dataset: {e}")

    def _setup_label_encoder(self):
        """Setup multi-label encoder with clean labels."""
        all_labels = set()
        
        # Collect all unique labels
        for label_list in self.labels:
            if isinstance(label_list, list):
                all_labels.update(label_list)
            else:
                all_labels.add(label_list)
        
        # Clean labels - remove any malformed entries
        clean_labels = set()
        for label in all_labels:
            if isinstance(label, str) and label.strip():
                clean_labels.add(label.strip())
        
        # Ensure we have default labels
        default_labels = {
            'COMFORT_EMOTIONAL', 'ENERGY_VITALITY', 'EXPERIMENTAL_FLAVOR',
            'HEALTH_LIGHT', 'WEATHER_HOT', 'WEATHER_COLD', 'SOCIAL_ROMANTIC',
            'SOCIAL_GROUP', 'SOCIAL_FAMILY', 'GENERAL_RECOMMENDATION'
        }
        clean_labels.update(default_labels)
        
        # Sort labels for consistency
        sorted_labels = sorted(list(clean_labels))
        
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


def improved_train_model(
    dish_dataset_path: str,
    output_dir: str = "models/improved_intent_classifier",
    num_epochs: int = 8,
    batch_size: int = 8,
    learning_rate: float = 2e-5
):
    """Improved model training with better configuration."""
    
    print("🚀 Starting improved model training...")
    
    # Create dataset
    dataset = ImprovedDishDataset(dish_dataset_path)
    
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
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(dataset.mlb.classes_),
        problem_type="multi_label_classification"
    )
    
    # Move model to device
    model.to(DEVICE)
    
    # Setup optimizer with better configuration
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=learning_rate * 0.1
    )
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    print("🏋️ Starting training...")
    
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.2e}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if os.path.exists(f"{output_dir}/best_model.pt"):
        model.load_state_dict(torch.load(f"{output_dir}/best_model.pt"))
        print("Loaded best model weights")
    
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
        'best_val_loss': best_val_loss,
        'final_train_loss': avg_loss,
        'final_val_loss': avg_val_loss,
    }
    
    with open(f"{output_dir}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print("✅ Training completed successfully!")
    print(f"📁 Model saved to: {output_dir}")
    print(f"🎯 Total labels: {len(dataset.mlb.classes_)}")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    
    return model, dataset


def test_improved_model(model_dir: str):
    """Test the improved model with sample queries."""
    print(f"🧪 Testing improved model from: {model_dir}")
    
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
        "Family dinner for kids",
        "Something warm for winter",
        "Adventurous and exotic food",
        "Low calorie healthy option",
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
            predictions = (probs > 0.3).cpu().numpy()[0]  # Lower threshold for better recall
            
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
    
    parser = argparse.ArgumentParser(description="Improved model training with dish dataset")
    parser.add_argument("--dish-dataset", type=str, default="data/processed/dish_dataset.parquet", 
                       help="Path to dish dataset")
    parser.add_argument("--output-dir", type=str, default="models/improved_intent_classifier",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--test-only", action="store_true", help="Only test existing model")
    
    args = parser.parse_args()
    
    if args.test_only:
        test_improved_model(args.output_dir)
    else:
        model, dataset = improved_train_model(
            dish_dataset_path=args.dish_dataset,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # Test the model
        print("\n" + "="*80)
        test_improved_model(args.output_dir)
