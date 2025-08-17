import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import numpy as np
import os
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ====== CONFIG ======
MODEL_DIR = "models/intent_classifier"
LABEL_BINARIZER_PATH = os.path.join(MODEL_DIR, "mlb_classes.json")
THRESHOLD = 0.5

# ====== LOAD ======
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Load MLB classes from JSON instead of pickle
with open(LABEL_BINARIZER_PATH, "r") as f:
    mlb_classes = json.load(f)

def apply_thresholds(logits, thresholds_path, top_k_fallback=3):
    """Apply per-label thresholds with fallback to avoid empty predictions."""
    probs = torch.sigmoid(logits).detach().cpu().numpy()[0]  # [C]
    
    # Load per-label thresholds if available
    if os.path.exists(thresholds_path):
        try:
            with open(thresholds_path, 'r') as f:
                thresholds = np.array(json.load(f))
            
            # Handle threshold mismatch: use thresholds for available classes, default for others
            if len(thresholds) == len(probs):
                preds = (probs >= thresholds).astype(int)
                logger.info(f"✅ Using tuned thresholds for {len(thresholds)} classes")
            else:
                logger.warning(f"Threshold mismatch: {len(thresholds)} thresholds vs {len(probs)} classes")
                # Use default threshold for all classes
                preds = (probs >= 0.5).astype(int)
                
        except Exception as e:
            logger.warning(f"Failed to load thresholds: {e}. Using default 0.5")
            preds = (probs >= 0.5).astype(int)
    else:
        logger.info("📁 No thresholds.json found, using default 0.5")
        preds = (probs >= 0.5).astype(int)
    
    # Fallback: if nothing fired, take top-k
    if preds.sum() == 0:
        logger.info(f"🔄 No predictions above threshold, using top-{top_k_fallback} fallback")
        top_idx = probs.argsort()[-top_k_fallback:][::-1]
        preds[top_idx] = 1
    
    return preds, probs

def filter_predictions(scores, labels, top_k=3, threshold=0.5):
    """Filter predictions by top_k and threshold for cleaner outputs."""
    preds = [(l, float(s)) for l, s in zip(labels, scores)]
    preds = sorted(preds, key=lambda x: x[1], reverse=True)
    return [l for l, s in preds[:top_k] if s >= threshold]

# ====== INFER FUNCTION ======
def predict_intents(text: str, top_k=3, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        
        # Use enhanced threshold system with fallbacks
        thresholds_path = os.path.join(MODEL_DIR, "thresholds.json")
        preds, probs = apply_thresholds(logits, thresholds_path, top_k_fallback=top_k)
        
        # Convert predictions to labels with smart filtering
        predicted_labels = []
        label_scores = []
        
        for i, pred in enumerate(preds):
            if pred == 1 and i < len(mlb_classes):
                label_scores.append((mlb_classes[i], probs[i]))
        
        # Sort by confidence and take top-k most confident predictions
        label_scores.sort(key=lambda x: x[1], reverse=True)
        predicted_labels = [label for label, score in label_scores[:top_k]]
        
        # Log the filtering process
        logger.info(f"📊 Filtered {len(label_scores)} predictions to top-{top_k}: {predicted_labels}")
    
    return predicted_labels, probs

# ====== EXAMPLE ======
if __name__ == "__main__":
    while True:
        text = input("\n🔍 Enter text (or type 'exit'): ").strip()
        if text.lower() == "exit":
            break

        labels, probs = predict_intents(text, top_k=3, threshold=0.5)
        print("✅ Predicted labels:", labels)

        # Show top predictions with scores
        print("📊 Top predictions:")
        for label, prob in zip(mlb_classes, probs):
            if prob >= 0.3:  # Show all predictions above 0.3 for debugging
                print(f"  {label:<25}: {prob:.3f}")