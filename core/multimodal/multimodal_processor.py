"""
Multi-Modal Input Processor
Phase 3: Advanced AI Features - Multi-modal Input Support

Supports:
- Text input (natural language)
- Image input (food photos, mood indicators)
- Voice input (speech-to-text)
- Combined multi-modal analysis
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoImageProcessor,
    pipeline
)
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import json
import logging
import base64
import io
import speech_recognition as sr
from dataclasses import dataclass
import cv2
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MultiModalInput:
    """Structured multi-modal input data."""
    text: Optional[str] = None
    image: Optional[Image.Image] = None
    audio: Optional[bytes] = None
    image_description: Optional[str] = None
    audio_transcript: Optional[str] = None

@dataclass
class MultiModalAnalysis:
    """Results of multi-modal analysis."""
    primary_mood: str
    confidence: float
    mood_categories: List[str]
    extracted_entities: List[str]
    image_analysis: Optional[Dict[str, Any]] = None
    audio_analysis: Optional[Dict[str, Any]] = None
    combined_confidence: float = 0.0

class MultiModalProcessor:
    """
    Advanced multi-modal processor for food recommendation system.
    
    Features:
    - Text processing with transformers
    - Image analysis (food recognition, mood indicators)
    - Speech-to-text processing
    - Multi-modal fusion for enhanced understanding
    """
    
    def __init__(
        self,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        image_model_name: str = "microsoft/resnet-50",
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name
        
        # Initialize models
        self._initialize_models()
        
        # Food-related image labels for analysis
        self.food_labels = [
            'food', 'meal', 'dish', 'cuisine', 'restaurant', 'cooking',
            'pizza', 'sushi', 'burger', 'pasta', 'salad', 'soup',
            'dessert', 'cake', 'ice cream', 'coffee', 'tea', 'drink'
        ]
        
        # Mood-indicating visual elements
        self.mood_visual_elements = {
            'comfort': ['warm', 'cozy', 'soft', 'familiar', 'home'],
            'excitement': ['bright', 'colorful', 'vibrant', 'energetic'],
            'romance': ['elegant', 'sophisticated', 'intimate', 'candlelight'],
            'health': ['fresh', 'green', 'natural', 'organic', 'clean'],
            'celebration': ['festive', 'decorated', 'party', 'sparkling']
        }
    
    def _initialize_models(self):
        """Initialize all required models."""
        logger.info("Initializing multi-modal models...")
        
        try:
            # Text processing models
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
            self.text_model = AutoModel.from_pretrained(self.text_model_name)
            
            # Image processing models
            self.image_processor = AutoImageProcessor.from_pretrained(self.image_model_name)
            self.image_model = AutoModel.from_pretrained(self.image_model_name)
            
            # Image classification pipeline
            self.image_classifier = pipeline(
                "image-classification",
                model="microsoft/resnet-50",
                device=0 if self.device == "cuda" else -1
            )
            
            # Image captioning pipeline
            self.image_captioner = pipeline(
                "image-to-text",
                model="nlpconnect/vit-gpt2-image-captioning",
                device=0 if self.device == "cuda" else -1
            )
            
            # Speech recognition
            self.speech_recognizer = sr.Recognizer()
            
            logger.info("Multi-modal models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process text input using transformer models.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text analysis results
        """
        if not text:
            return {}
        
        # Tokenize and encode text
        inputs = self.text_tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        # Get text embeddings
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Pooling
        
        return {
            'text': text,
            'embeddings': embeddings.cpu().numpy(),
            'length': len(text),
            'processed': True
        }
    
    def process_image(self, image: Union[Image.Image, str, bytes]) -> Dict[str, Any]:
        """
        Process image input for food and mood analysis.
        
        Args:
            image: PIL Image, base64 string, or bytes
            
        Returns:
            Dictionary with image analysis results
        """
        if not image:
            return {}
        
        # Convert to PIL Image if needed
        if isinstance(image, str):
            # Assume base64 encoded image
            try:
                image_data = base64.b64decode(image)
                image = Image.open(io.BytesIO(image_data))
            except:
                logger.error("Invalid base64 image string")
                return {}
        
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        
        if not isinstance(image, Image.Image):
            logger.error("Invalid image format")
            return {}
        
        try:
            # Image classification
            classification_results = self.image_classifier(image)
            
            # Image captioning
            caption_results = self.image_captioner(image)
            caption = caption_results[0]['generated_text'] if caption_results else ""
            
            # Analyze for food-related content
            food_confidence = self._analyze_food_content(classification_results)
            
            # Analyze for mood indicators
            mood_indicators = self._analyze_mood_indicators(caption, classification_results)
            
            return {
                'image': image,
                'caption': caption,
                'classification': classification_results,
                'food_confidence': food_confidence,
                'mood_indicators': mood_indicators,
                'processed': True
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {}
    
    def process_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Process audio input using speech recognition.
        
        Args:
            audio_data: Audio bytes (WAV format preferred)
            
        Returns:
            Dictionary with audio analysis results
        """
        if not audio_data:
            return {}
        
        try:
            # Convert bytes to audio data
            audio = sr.AudioData(audio_data, sample_rate=16000, sample_width=2)
            
            # Speech recognition
            transcript = self.speech_recognizer.recognize_google(audio)
            
            # Process the transcript as text
            text_analysis = self.process_text(transcript)
            
            return {
                'audio': audio_data,
                'transcript': transcript,
                'text_analysis': text_analysis,
                'processed': True
            }
            
        except sr.UnknownValueError:
            logger.warning("Speech recognition could not understand audio")
            return {}
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return {}
    
    def _analyze_food_content(self, classification_results: List[Dict]) -> float:
        """Analyze if image contains food-related content."""
        food_confidence = 0.0
        
        for result in classification_results:
            label = result['label'].lower()
            confidence = result['score']
            
            # Check if label is food-related
            for food_label in self.food_labels:
                if food_label in label:
                    food_confidence = max(food_confidence, confidence)
                    break
        
        return food_confidence
    
    def _analyze_mood_indicators(self, caption: str, classification_results: List[Dict]) -> Dict[str, float]:
        """Analyze image for mood-indicating visual elements."""
        mood_scores = {mood: 0.0 for mood in self.mood_visual_elements.keys()}
        
        # Analyze caption
        caption_lower = caption.lower()
        for mood, indicators in self.mood_visual_elements.items():
            for indicator in indicators:
                if indicator in caption_lower:
                    mood_scores[mood] += 0.3
        
        # Analyze classification results
        for result in classification_results:
            label = result['label'].lower()
            confidence = result['score']
            
            for mood, indicators in self.mood_visual_elements.items():
                for indicator in indicators:
                    if indicator in label:
                        mood_scores[mood] = max(mood_scores[mood], confidence * 0.5)
        
        return mood_scores
    
    def process_multimodal(
        self, 
        text: Optional[str] = None,
        image: Optional[Union[Image.Image, str, bytes]] = None,
        audio: Optional[bytes] = None
    ) -> MultiModalAnalysis:
        """
        Process multi-modal input and return combined analysis.
        
        Args:
            text: Text input
            image: Image input
            audio: Audio input
            
        Returns:
            MultiModalAnalysis with combined results
        """
        # Process each modality
        text_analysis = self.process_text(text) if text else {}
        image_analysis = self.process_image(image) if image else {}
        audio_analysis = self.process_audio(audio) if audio else {}
        
        # Combine results
        combined_text = text or ""
        if audio_analysis.get('transcript'):
            combined_text += " " + audio_analysis['transcript']
        
        # Extract mood categories from text
        mood_categories = self._extract_mood_categories(combined_text)
        
        # Add mood indicators from image
        if image_analysis.get('mood_indicators'):
            image_moods = [
                mood for mood, score in image_analysis['mood_indicators'].items()
                if score > 0.3
            ]
            mood_categories.extend(image_moods)
        
        # Remove duplicates and get primary mood
        mood_categories = list(set(mood_categories))
        primary_mood = mood_categories[0] if mood_categories else "general"
        
        # Calculate combined confidence
        text_confidence = 0.8 if text_analysis else 0.0
        image_confidence = image_analysis.get('food_confidence', 0.0)
        audio_confidence = 0.7 if audio_analysis.get('transcript') else 0.0
        
        # Weighted combination
        total_weight = (1.0 if text_analysis else 0.0) + (0.8 if image_analysis else 0.0) + (0.6 if audio_analysis else 0.0)
        combined_confidence = (
            text_confidence * (1.0 if text_analysis else 0.0) +
            image_confidence * 0.8 * (1.0 if image_analysis else 0.0) +
            audio_confidence * 0.6 * (1.0 if audio_analysis else 0.0)
        ) / total_weight if total_weight > 0 else 0.0
        
        # Extract entities
        entities = self._extract_entities(combined_text)
        
        return MultiModalAnalysis(
            primary_mood=primary_mood,
            confidence=combined_confidence,
            mood_categories=mood_categories,
            extracted_entities=entities,
            image_analysis=image_analysis if image_analysis else None,
            audio_analysis=audio_analysis if audio_analysis else None,
            combined_confidence=combined_confidence
        )
    
    def _extract_mood_categories(self, text: str) -> List[str]:
        """Extract mood categories from text."""
        text_lower = text.lower()
        mood_categories = []
        
        # Simple keyword-based extraction (can be enhanced with ML)
        mood_keywords = {
            'comfort': ['comfort', 'warm', 'cozy', 'soothing', 'nurturing'],
            'excitement': ['exciting', 'spicy', 'bold', 'energetic', 'vibrant'],
            'romance': ['romantic', 'elegant', 'sophisticated', 'intimate'],
            'health': ['healthy', 'fresh', 'light', 'clean', 'natural'],
            'celebration': ['celebration', 'party', 'festive', 'special'],
            'weather_hot': ['hot', 'summer', 'refreshing', 'cool'],
            'weather_cold': ['cold', 'winter', 'warm', 'hearty'],
            'energy_light': ['light', 'easy', 'gentle', 'subtle'],
            'energy_heavy': ['heavy', 'filling', 'substantial', 'rich']
        }
        
        for mood, keywords in mood_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                mood_categories.append(mood)
        
        return mood_categories
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text."""
        entities = []
        text_lower = text.lower()
        
        # Entity patterns
        entity_patterns = {
            'weather': ['hot', 'cold', 'warm', 'cool', 'sunny', 'rainy'],
            'time': ['morning', 'afternoon', 'evening', 'night'],
            'social': ['alone', 'couple', 'family', 'friends'],
            'flavor': ['sweet', 'spicy', 'salty', 'savory'],
            'energy': ['light', 'heavy', 'greasy', 'filling']
        }
        
        for category, patterns in entity_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    entities.append(pattern)
        
        return entities
    
    def get_processing_info(self) -> Dict[str, Any]:
        """Get information about the multi-modal processor."""
        return {
            'text_model': self.text_model_name,
            'image_model': self.image_model_name,
            'device': self.device,
            'food_labels_count': len(self.food_labels),
            'mood_elements_count': len(self.mood_visual_elements),
            'models_loaded': {
                'text': self.text_model is not None,
                'image': self.image_model is not None,
                'speech': self.speech_recognizer is not None
            }
        }

# Convenience functions
def process_text_input(text: str) -> Dict[str, Any]:
    """Process text input only."""
    processor = MultiModalProcessor()
    return processor.process_text(text)

def process_image_input(image: Union[Image.Image, str, bytes]) -> Dict[str, Any]:
    """Process image input only."""
    processor = MultiModalProcessor()
    return processor.process_image(image)

def process_audio_input(audio_data: bytes) -> Dict[str, Any]:
    """Process audio input only."""
    processor = MultiModalProcessor()
    return processor.process_audio(audio_data)

def process_combined_input(
    text: Optional[str] = None,
    image: Optional[Union[Image.Image, str, bytes]] = None,
    audio: Optional[bytes] = None
) -> MultiModalAnalysis:
    """Process combined multi-modal input."""
    processor = MultiModalProcessor()
    return processor.process_multimodal(text=text, image=image, audio=audio) 