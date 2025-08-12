"""
Enhanced Intent Classifier with Transformers and Semantic Embeddings
Phase 3: Advanced AI Features - Deep Learning Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json
import pickle
from pathlib import Path
import logging
from sklearn.metrics.pairwise import cosine_similarity
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
    - Transformer-based text understanding
    - Semantic embeddings for food-mood relationships
    - Multi-label classification
    - Confidence scoring
    - Real-time learning capabilities
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        taxonomy_path: str = "data/taxonomy/mood_food_taxonomy.json",
        model_save_path: str = "models/enhanced_intent_classifier",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.taxonomy_path = taxonomy_path
        self.model_save_path = Path(model_save_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load taxonomy
        self.taxonomy = self._load_taxonomy()
        self.intent_labels = self._extract_intent_labels()
        
        # Initialize models
        self.sentence_transformer = None
        self.intent_classifier = None
        self.intent_embeddings = {}
        
        # Load or initialize models
        self._initialize_models()
        
        # Learning components
        self.feedback_buffer = []
        self.learning_rate = 0.001
        self.min_feedback_threshold = 10
        
    def _load_taxonomy(self) -> Dict[str, Any]:
        """Load the mood-food taxonomy."""
        with open(self.taxonomy_path, 'r') as f:
            return json.load(f)
    
    def _extract_intent_labels(self) -> List[str]:
        """Extract all unique intent labels from taxonomy."""
        labels = set()
        for category, data in self.taxonomy.items():
            if 'labels' in data:
                labels.update(data['labels'])
            if 'descriptors' in data:
                # Add category as a label too
                labels.add(category)
        return sorted(list(labels))
    
    def _initialize_models(self):
        """Initialize or load the transformer models."""
        try:
            # Try to load existing models
            if self.model_save_path.exists():
                logger.info("Loading existing enhanced intent classifier...")
                self._load_models()
            else:
                logger.info("Initializing new enhanced intent classifier...")
                self._create_models()
        except Exception as e:
            logger.warning(f"Error loading models: {e}. Creating new ones...")
            self._create_models()
    
    def _create_models(self):
        """Create new transformer models."""
        # Initialize sentence transformer for semantic embeddings
        self.sentence_transformer = SentenceTransformer(self.model_name)
        
        # Create intent embeddings from taxonomy
        self._create_intent_embeddings()
        
        # Save models
        self._save_models()
    
    def _create_intent_embeddings(self):
        """Create semantic embeddings for all intent labels."""
        logger.info("Creating semantic embeddings for intent labels...")
        
        # Create embeddings for each intent label
        for label in self.intent_labels:
            # Use the label itself and related descriptors for embedding
            embedding_texts = [label]
            
            # Add related descriptors from taxonomy
            for category, data in self.taxonomy.items():
                if label in data.get('labels', []):
                    embedding_texts.extend(data.get('descriptors', []))
            
            # Create embedding from combined text
            combined_text = " ".join(embedding_texts)
            embedding = self.sentence_transformer.encode(combined_text, convert_to_tensor=True)
            self.intent_embeddings[label] = embedding
        
        logger.info(f"Created embeddings for {len(self.intent_embeddings)} intent labels")
    
    def _save_models(self):
        """Save the trained models and embeddings."""
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # Save sentence transformer
        self.sentence_transformer.save(str(self.model_save_path / "sentence_transformer"))
        
        # Save intent embeddings
        embeddings_dict = {
            label: embedding.cpu().numpy() 
            for label, embedding in self.intent_embeddings.items()
        }
        with open(self.model_save_path / "intent_embeddings.pkl", 'wb') as f:
            pickle.dump(embeddings_dict, f)
        
        # Save intent labels
        with open(self.model_save_path / "intent_labels.json", 'w') as f:
            json.dump(self.intent_labels, f)
        
        logger.info(f"Models saved to {self.model_save_path}")
    
    def _load_models(self):
        """Load existing trained models."""
        # Load sentence transformer
        self.sentence_transformer = SentenceTransformer(str(self.model_save_path / "sentence_transformer"))
        
        # Load intent embeddings
        with open(self.model_save_path / "intent_embeddings.pkl", 'rb') as f:
            embeddings_dict = pickle.load(f)
        
        self.intent_embeddings = {
            label: torch.tensor(embedding, device=self.device)
            for label, embedding in embeddings_dict.items()
        }
        
        # Load intent labels
        with open(self.model_save_path / "intent_labels.json", 'r') as f:
            self.intent_labels = json.load(f)
        
        logger.info("Models loaded successfully")
    
    def classify_intent(self, text: str, top_k: int = 5) -> IntentPrediction:
        """
        Classify user intent using semantic similarity.
        
        Args:
            text: User input text
            top_k: Number of top intents to return
            
        Returns:
            IntentPrediction with primary intent and confidence scores
        """
        # Encode user input
        user_embedding = self.sentence_transformer.encode(text, convert_to_tensor=True)
        
        # Calculate similarities with all intent embeddings
        similarities = {}
        for label, intent_embedding in self.intent_embeddings.items():
            similarity = util.pytorch_cos_sim(user_embedding, intent_embedding).item()
            similarities[label] = similarity
        
        # Sort by similarity
        sorted_intents = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Get top-k intents
        top_intents = sorted_intents[:top_k]
        
        # Primary intent is the one with highest similarity
        primary_intent, primary_confidence = top_intents[0]
        
        return IntentPrediction(
            primary_intent=primary_intent,
            confidence=primary_confidence,
            all_intents=top_intents,
            embeddings=user_embedding.cpu().numpy(),
            semantic_similarity=primary_confidence
        )
    
    def get_mood_categories(self, text: str, threshold: float = 0.3) -> List[str]:
        """
        Get mood categories based on intent classification.
        
        Args:
            text: User input text
            threshold: Minimum confidence threshold
            
        Returns:
            List of mood categories
        """
        prediction = self.classify_intent(text)
        
        # Filter intents above threshold
        valid_intents = [
            intent for intent, confidence in prediction.all_intents 
            if confidence >= threshold
        ]
        
        # Map intents to mood categories
        mood_categories = []
        for intent in valid_intents:
            # Direct category match
            if intent in self.taxonomy:
                mood_categories.append(intent)
            else:
                # Find categories that contain this intent in their labels
                for category, data in self.taxonomy.items():
                    if intent in data.get('labels', []):
                        mood_categories.append(category)
        
        return list(set(mood_categories))  # Remove duplicates
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract entities from text using semantic similarity.
        
        Args:
            text: User input text
            
        Returns:
            List of extracted entities
        """
        # Define entity categories
        entity_categories = {
            'weather': ['hot', 'cold', 'warm', 'cool', 'sunny', 'rainy', 'snowy', 'summer', 'winter'],
            'emotions': ['sad', 'happy', 'stressed', 'anxious', 'excited', 'tired', 'energetic'],
            'time': ['morning', 'afternoon', 'evening', 'night', 'breakfast', 'lunch', 'dinner'],
            'social': ['alone', 'couple', 'family', 'friends', 'party', 'date'],
            'energy': ['light', 'heavy', 'greasy', 'filling', 'substantial'],
            'flavor': ['sweet', 'spicy', 'salty', 'savory', 'bitter'],
            'health': ['sick', 'recovery', 'healthy', 'gentle', 'nourishing']
        }
        
        # Encode user text
        user_embedding = self.sentence_transformer.encode(text, convert_to_tensor=True)
        
        extracted_entities = []
        
        for category, entities in entity_categories.items():
            for entity in entities:
                if entity.lower() in text.lower():
                    # Calculate semantic similarity for additional confidence
                    entity_embedding = self.sentence_transformer.encode(entity, convert_to_tensor=True)
                    similarity = util.pytorch_cos_sim(user_embedding, entity_embedding).item()
                    
                    if similarity > 0.5:  # Threshold for semantic similarity
                        extracted_entities.append(entity)
        
        return extracted_entities
    
    def update_with_feedback(self, text: str, correct_intents: List[str], confidence: float = 1.0):
        """
        Update the model with user feedback for real-time learning.
        
        Args:
            text: Original user input
            correct_intents: List of correct intent labels
            confidence: Confidence in the feedback (0-1)
        """
        # Store feedback for batch learning
        self.feedback_buffer.append({
            'text': text,
            'correct_intents': correct_intents,
            'confidence': confidence,
            'timestamp': torch.tensor(time.time())
        })
        
        # Check if we have enough feedback for learning
        if len(self.feedback_buffer) >= self.min_feedback_threshold:
            self._perform_online_learning()
    
    def _perform_online_learning(self):
        """Perform online learning with accumulated feedback."""
        logger.info(f"Performing online learning with {len(self.feedback_buffer)} feedback samples")
        
        # Simple online learning: update embeddings based on feedback
        for feedback in self.feedback_buffer:
            text = feedback['text']
            correct_intents = feedback['correct_intents']
            confidence = feedback['confidence']
            
            # Get current prediction
            current_prediction = self.classify_intent(text)
            
            # Update embeddings for correct intents
            text_embedding = self.sentence_transformer.encode(text, convert_to_tensor=True)
            
            for intent in correct_intents:
                if intent in self.intent_embeddings:
                    # Move intent embedding closer to text embedding
                    current_embedding = self.intent_embeddings[intent]
                    updated_embedding = current_embedding + self.learning_rate * confidence * (text_embedding - current_embedding)
                    self.intent_embeddings[intent] = F.normalize(updated_embedding, p=2, dim=0)
        
        # Clear feedback buffer
        self.feedback_buffer = []
        
        # Save updated models
        self._save_models()
        
        logger.info("Online learning completed and models saved")
    
    def get_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        embedding1 = self.sentence_transformer.encode(text1, convert_to_tensor=True)
        embedding2 = self.sentence_transformer.encode(text2, convert_to_tensor=True)
        
        similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
        return similarity
    
    def find_similar_foods(self, food_name: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find semantically similar foods.
        
        Args:
            food_name: Name of the food
            top_k: Number of similar foods to return
            
        Returns:
            List of (food_name, similarity_score) tuples
        """
        food_embedding = self.sentence_transformer.encode(food_name, convert_to_tensor=True)
        
        similarities = []
        for category, data in self.taxonomy.items():
            for food in data.get('foods', []):
                food_desc = f"{food['name']} {food.get('culture', '')} {food.get('region', '')}"
                food_embedding_other = self.sentence_transformer.encode(food_desc, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(food_embedding, food_embedding_other).item()
                similarities.append((food['name'], similarity))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'num_intent_labels': len(self.intent_labels),
            'num_embeddings': len(self.intent_embeddings),
            'feedback_buffer_size': len(self.feedback_buffer),
            'learning_rate': self.learning_rate,
            'model_save_path': str(self.model_save_path)
        }

# Convenience function for easy usage
def create_enhanced_classifier(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    taxonomy_path: str = "data/taxonomy/mood_food_taxonomy.json"
) -> EnhancedIntentClassifier:
    """Create and return an enhanced intent classifier."""
    return EnhancedIntentClassifier(model_name=model_name, taxonomy_path=taxonomy_path) 