"""
Enhanced Intent Classifier with Transformers and Semantic Embeddings
Phase 3: Advanced AI Features - Deep Learning Models
"""

import torch
import torch.nn.functional as F
try:
    from sentence_transformers import SentenceTransformer, util  # type: ignore
    _SENTENCE_TX_AVAILABLE = True
except Exception:
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore
    _SENTENCE_TX_AVAILABLE = False
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json
import pickle
from pathlib import Path
import logging
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IntentPrediction:
    """Structured prediction result with confidence scores."""
    primary_intent: str
    confidence: float
    all_intents: List[Tuple[str, float]]
    embeddings: Optional[np.ndarray] = None
    semantic_similarity: Optional[float] = None

class EnhancedIntentClassifier:
    """
    Advanced intent classifier using transformers and semantic embeddings.

    Features:
    - Transformer-based text understanding (lazy-loaded)
    - Semantic embeddings for food-mood relationships (lazy-built)
    - Multi-label classification
    - Confidence scoring
    - Real-time learning capabilities
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        taxonomy_path: str = "data/taxonomy/mood_food_taxonomy.json",
        model_save_path: str = "models/enhanced_intent_classifier",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.taxonomy_path = taxonomy_path
        self.model_save_path = Path(model_save_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load taxonomy and labels
        self.taxonomy = self._load_taxonomy()
        self.intent_labels = self._extract_intent_labels()

        # Lazy state
        self.sentence_transformer = None
        self.intent_embeddings: Dict[str, torch.Tensor] = {}
        self._models_loaded = False

        # Learning components
        self.feedback_buffer: List[Dict[str, Any]] = []
        self.learning_rate = 0.001
        self.min_feedback_threshold = 10

    # -------- Model/Embeddings Management --------
    def _ensure_models(self):
        if not _SENTENCE_TX_AVAILABLE:
            raise RuntimeError("sentence-transformers not installed. Install requirements_phase3.txt")
        if not self._models_loaded:
            # Try load from disk first
            try:
                if (self.model_save_path / "sentence_transformer").exists():
                    self._load_models()
                    self._models_loaded = True
                    logger.info("Enhanced intent classifier models loaded from disk")
                    return
            except Exception as e:
                logger.warning(f"Failed loading saved models: {e}")
            # Fresh init
            logger.info("Initializing sentence transformer (lazy)... This may download weights on first run.")
            self.sentence_transformer = SentenceTransformer(self.model_name)
            # Build embeddings for labels
            self._create_intent_embeddings()
            self._save_models()
            self._models_loaded = True

    def _load_taxonomy(self) -> Dict[str, Any]:
        with open(self.taxonomy_path, "r") as f:
            return json.load(f)

    def _extract_intent_labels(self) -> List[str]:
        labels = set()
        for category, data in self.taxonomy.items():
            if "labels" in data:
                labels.update(data["labels"])
            # include category token as a label seed
            labels.add(category)
        return sorted(labels)

    def _create_intent_embeddings(self):
        assert self.sentence_transformer is not None
        logger.info("Creating semantic embeddings for intent labels (lazy)...")
        for label in self.intent_labels:
            embedding_texts = [label]
            for _, data in self.taxonomy.items():
                if label in data.get("labels", []):
                    embedding_texts.extend(data.get("descriptors", []))
            combined_text = " ".join(embedding_texts)
            emb = self.sentence_transformer.encode(combined_text, convert_to_tensor=True)
            self.intent_embeddings[label] = emb
        logger.info(f"Created embeddings for {len(self.intent_embeddings)} labels")

    def _save_models(self):
        try:
            self.model_save_path.mkdir(parents=True, exist_ok=True)
            if self.sentence_transformer is not None:
                self.sentence_transformer.save(str(self.model_save_path / "sentence_transformer"))
            embeddings_dict = {k: v.cpu().numpy() for k, v in self.intent_embeddings.items()}
            with open(self.model_save_path / "intent_embeddings.pkl", "wb") as f:
                pickle.dump(embeddings_dict, f)
            with open(self.model_save_path / "intent_labels.json", "w") as f:
                json.dump(self.intent_labels, f)
        except Exception as e:
            logger.warning(f"Failed saving enhanced models: {e}")

    def _load_models(self):
        assert _SENTENCE_TX_AVAILABLE
        from sentence_transformers import SentenceTransformer as _ST  # local import
        self.sentence_transformer = _ST(str(self.model_save_path / "sentence_transformer"))
        with open(self.model_save_path / "intent_embeddings.pkl", "rb") as f:
            embeddings_dict = pickle.load(f)
        self.intent_embeddings = {k: torch.tensor(v, device=self.device) for k, v in embeddings_dict.items()}
        with open(self.model_save_path / "intent_labels.json", "r") as f:
            self.intent_labels = json.load(f)

    # -------- Inference API --------
    def classify_intent(self, text: str, top_k: int = 5) -> IntentPrediction:
        self._ensure_models()
        assert self.sentence_transformer is not None and util is not None
        user_embedding = self.sentence_transformer.encode(text, convert_to_tensor=True)
        similarities: List[Tuple[str, float]] = []
        for label, intent_embedding in self.intent_embeddings.items():
            sim = util.pytorch_cos_sim(user_embedding, intent_embedding).item()
            similarities.append((label, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_intents = similarities[:top_k]
        primary_intent, primary_confidence = top_intents[0]
        return IntentPrediction(
            primary_intent=primary_intent,
            confidence=primary_confidence,
            all_intents=top_intents,
            embeddings=user_embedding.detach().cpu().numpy(),
            semantic_similarity=primary_confidence,
        )

    def get_mood_categories(self, text: str, threshold: float = 0.3) -> List[str]:
        pred = self.classify_intent(text)
        valid_intents = [i for i, c in pred.all_intents if c >= threshold]
        mood_categories: List[str] = []
        for intent in valid_intents:
            if intent in self.taxonomy:
                mood_categories.append(intent)
            else:
                for category, data in self.taxonomy.items():
                    if intent in data.get("labels", []):
                        mood_categories.append(category)
        return list(set(mood_categories))

    def extract_entities(self, text: str) -> List[str]:
        if not _SENTENCE_TX_AVAILABLE:
            return []
        assert self.sentence_transformer is not None or self._models_loaded is False
        if self.sentence_transformer is None:
            # Light fallback: keyword scan until model is ready
            keywords = [
                "hot","cold","warm","cool","sunny","rainy","sad","happy","stressed","excited",
                "morning","afternoon","evening","night","alone","couple","family","friends","party",
                "light","heavy","greasy","filling","sweet","spicy","salty","savory","bitter","healthy"
            ]
            return [k for k in keywords if k in text.lower()]
        # Semantic scoring (optional)
        entities_found: List[str] = []
        for token in set(text.lower().split()):
            if len(token) < 3:
                continue
            try:
                score = self.get_semantic_similarity(text, token)
                if score > 0.6:
                    entities_found.append(token)
            except Exception:
                continue
        return entities_found[:10]

    def update_with_feedback(self, text: str, correct_intents: List[str], confidence: float = 1.0):
        self._ensure_models()
        text_embedding = self.sentence_transformer.encode(text, convert_to_tensor=True)  # type: ignore
        self.feedback_buffer.append({
            "text": text,
            "correct_intents": correct_intents,
            "confidence": confidence,
            "timestamp": time.time(),
        })
        for intent in correct_intents:
            if intent in self.intent_embeddings:
                current_embedding = self.intent_embeddings[intent]
                updated_embedding = current_embedding + self.learning_rate * confidence * (text_embedding - current_embedding)
                self.intent_embeddings[intent] = F.normalize(updated_embedding, p=2, dim=0)
        if len(self.feedback_buffer) >= self.min_feedback_threshold:
            self._save_models()
            self.feedback_buffer = []

    def get_semantic_similarity(self, text1: str, text2: str) -> float:
        self._ensure_models()
        assert self.sentence_transformer is not None and util is not None
        e1 = self.sentence_transformer.encode(text1, convert_to_tensor=True)
        e2 = self.sentence_transformer.encode(text2, convert_to_tensor=True)
        return util.pytorch_cos_sim(e1, e2).item()

    def find_similar_foods(self, food_name: str, top_k: int = 5) -> List[Tuple[str, float]]:
        self._ensure_models()
        assert self.sentence_transformer is not None and util is not None
        food_emb = self.sentence_transformer.encode(food_name, convert_to_tensor=True)
        similarities: List[Tuple[str, float]] = []
        for _, data in self.taxonomy.items():
            for food in data.get("foods", []):
                desc = f"{food['name']} {food.get('culture','')} {food.get('region','')}"
                other_emb = self.sentence_transformer.encode(desc, convert_to_tensor=True)
                sim = util.pytorch_cos_sim(food_emb, other_emb).item()
                similarities.append((food["name"], sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "num_intent_labels": len(self.intent_labels),
            "num_embeddings": len(self.intent_embeddings),
            "feedback_buffer_size": len(self.feedback_buffer),
            "learning_rate": self.learning_rate,
            "model_save_path": str(self.model_save_path),
            "loaded": self._models_loaded,
            "sentence_transformers_available": _SENTENCE_TX_AVAILABLE,
        }

# Convenience

def create_enhanced_classifier(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    taxonomy_path: str = "data/taxonomy/mood_food_taxonomy.json",
) -> EnhancedIntentClassifier:
    return EnhancedIntentClassifier(model_name=model_name, taxonomy_path=taxonomy_path) 