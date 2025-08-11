from core.nlu.entity_extraction import extract_entities
from core.nlu.intent_classification import classify_intent

def test_basic_intent_and_entities():
    text = "I want something light and refreshing for lunch with friends"
    intent_result = classify_intent(text)
    entities = extract_entities(text)

    assert "HEALTH_WELLNESS" in intent_result["primary_intent"] or "OCCASION_SOCIAL" in intent_result["primary_intent"]
    assert "lunch" in entities["MEAL_TYPE"] or "lunch" in text.lower()
    assert "with_friends" in entities["SOCIAL_CONTEXT"] or "with friends" in text.lower()