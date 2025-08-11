import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import numpy as np
import os

# ====== CONFIG ======
MODEL_DIR = "models/intent"
LABEL_BINARIZER_PATH = os.path.join(MODEL_DIR, "label_binarizer.pkl")
THRESHOLD = 0.5

# ====== LOAD ======
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

mlb = joblib.load(LABEL_BINARIZER_PATH)

# ====== INFER FUNCTION ======
def predict_intents(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        preds = (probs >= THRESHOLD).astype(int)

    predicted_labels = mlb.inverse_transform([preds])[0]
    return predicted_labels, probs

# ====== EXAMPLE ======
if __name__ == "__main__":
    while True:
        text = input("\n🔍 Enter text (or type 'exit'): ").strip()
        if text.lower() == "exit":
            break

        labels, probs = predict_intents(text)
        print("✅ Predicted labels:", labels)

        for label, prob in zip(mlb.classes_, probs):
            print(f"  {label:<25}: {prob:.2f}")