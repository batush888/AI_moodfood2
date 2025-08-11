import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.routes:app", host="0.0.0.0", port=8000, reload=True)

from core.nlu.infer_intent import IntentClassifier

classifier = IntentClassifier()
intent, confidence = classifier.predict("I want something warm")

from core.nlu.inference import classify_intent
from data.taxonomy.loader import load_taxonomy

taxonomy = load_taxonomy()

user_input = "I want something comforting"
intent = classify_intent(user_input)
foods = taxonomy[intent]["foods"]
print(f"Suggested for '{intent}': {foods}")