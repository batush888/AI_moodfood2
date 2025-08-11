from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

class IntentClassifier:
    def __init__(self, model_path="models/intent_model"):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.label_encoder = torch.load("models/label_encoder.pt")
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            label = self.label_encoder.inverse_transform([pred_idx])[0]
            return label, probs[0][pred_idx].item()
        
from data.taxonomy.loader import load_taxonomy

taxonomy = load_taxonomy()
intent = classify_intent(user_input)
suggested_foods = taxonomy[intent]["foods"]