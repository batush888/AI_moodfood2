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
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoImageProcessor,
    pipeline,
)
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import json
import logging
import base64
import io
from dataclasses import dataclass
from pathlib import Path

# Optional dependency: SpeechRecognition
try:
    import speech_recognition as sr  # type: ignore
except Exception:  # pragma: no cover
    sr = None

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
    - Text processing with transformers (lazy init)
    - Image analysis (food recognition, mood indicators) (lazy init)
    - Speech-to-text processing (if available) (lazy init)
    - Multi-modal fusion for enhanced understanding
    """

    def __init__(
        self,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        image_model_name: str = "microsoft/resnet-50",
        device: Optional[str] = None,
    ):
        # Enhanced device detection for Apple Silicon
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name

        # Lazy-initialized members
        self.text_tokenizer = None
        self.text_model = None
        self.image_processor = None
        self.image_model = None
        self.image_classifier = None
        self.image_captioner = None
        self.speech_recognizer = None

        # Food-related image labels for analysis
        self.food_labels = [
            "food",
            "meal",
            "dish",
            "cuisine",
            "restaurant",
            "cooking",
            "pizza",
            "sushi",
            "burger",
            "pasta",
            "salad",
            "soup",
            "dessert",
            "cake",
            "ice cream",
            "coffee",
            "tea",
            "drink",
        ]

        # Mood-indicating visual elements
        self.mood_visual_elements = {
            "comfort": ["warm", "cozy", "soft", "familiar", "home"],
            "excitement": ["bright", "colorful", "vibrant", "energetic"],
            "romance": ["elegant", "sophisticated", "intimate", "candlelight"],
            "health": ["fresh", "green", "natural", "organic", "clean"],
            "celebration": ["festive", "decorated", "party", "sparkling"],
        }

    def _ensure_text_models(self):
        if self.text_tokenizer is None or self.text_model is None:
            logger.info("Initializing text models for multi-modal processor (lazy)...")
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
            self.text_model = AutoModel.from_pretrained(self.text_model_name)
            logger.info("Text models ready")

    def _ensure_image_models(self):
        if self.image_classifier is None or self.image_captioner is None:
            logger.info("Initializing image pipelines for multi-modal processor (lazy)...")
            try:
                # Enhanced device mapping for different platforms
                device_id = 0 if self.device == "cuda" else -1
                if self.device == "mps":
                    device_id = -1  # MPS not fully supported by all pipelines yet
                
                # Pipelines handle their own processors/models internally
                self.image_classifier = pipeline(
                    "image-classification",
                    model=self.image_model_name,
                    device=device_id,
                )
                self.image_captioner = pipeline(
                    "image-to-text",
                    model="nlpconnect/vit-gpt2-image-captioning",
                    device=device_id,
                )
                logger.info("Image pipelines ready")
            except Exception as e:
                logger.error(f"Error initializing image models: {e}")
                # Fallback to basic image processing
                self.image_classifier = None
                self.image_captioner = None

    def _ensure_speech_models(self):
        if self.speech_recognizer is None:
            if sr is None:
                logger.warning("SpeechRecognition not available; skipping audio processing")
            else:
                logger.info("Initializing speech recognizer (lazy)...")
                self.speech_recognizer = sr.Recognizer()

    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process text input using transformer models.
        """
        if not text:
            return {}

        self._ensure_text_models()

        inputs = self.text_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = self.text_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Pooling

        return {
            "text": text,
            "embeddings": embeddings.cpu().numpy(),
            "length": len(text),
            "processed": True,
        }

    def process_image(self, image: Union[Image.Image, str, bytes]) -> Dict[str, Any]:
        """
        Process image input for food and mood analysis.
        """
        if not image:
            return {}

        # Convert to PIL Image if needed
        if isinstance(image, str):
            try:
                image_data = base64.b64decode(image)
                image = Image.open(io.BytesIO(image_data))
            except Exception:
                logger.error("Invalid base64 image string")
                return {}
        elif isinstance(image, bytes):
            try:
                image = Image.open(io.BytesIO(image))
            except Exception:
                logger.error("Invalid image bytes")
                return {}

        if not isinstance(image, Image.Image):
            logger.error("Invalid image format")
            return {}

        try:
            self._ensure_image_models()

            classification_results = self.image_classifier(image)
            caption_results = self.image_captioner(image)
            caption = (
                caption_results[0]["generated_text"] if caption_results else ""
            )

            food_confidence = self._analyze_food_content(classification_results)
            mood_indicators = self._analyze_mood_indicators(
                caption, classification_results
            )

            return {
                "caption": caption,
                "classification": classification_results,
                "food_confidence": food_confidence,
                "mood_indicators": mood_indicators,
                "processed": True,
            }

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {}

    def process_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Process audio input using speech recognition.
        """
        if not audio_data:
            return {}

        self._ensure_speech_models()
        if self.speech_recognizer is None:
            return {}

        try:
            # Assume 16kHz mono 16-bit PCM WAV bytes
            audio = sr.AudioData(audio_data, sample_rate=16000, sample_width=2)
            transcript = self.speech_recognizer.recognize_google(audio)
            text_analysis = self.process_text(transcript)

            return {
                "transcript": transcript,
                "text_analysis": text_analysis,
                "processed": True,
            }

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return {}

    def _analyze_food_content(self, classification_results: List[Dict]) -> float:
        food_confidence = 0.0
        for result in classification_results:
            label = result.get("label", "").lower()
            confidence = float(result.get("score", 0.0))
            if any(food_label in label for food_label in self.food_labels):
                food_confidence = max(food_confidence, confidence)
        return food_confidence

    def _analyze_mood_indicators(
        self, caption: str, classification_results: List[Dict]
    ) -> Dict[str, float]:
        mood_scores = {mood: 0.0 for mood in self.mood_visual_elements.keys()}

        caption_lower = caption.lower()
        for mood, indicators in self.mood_visual_elements.items():
            for indicator in indicators:
                if indicator in caption_lower:
                    mood_scores[mood] += 0.3

        for result in classification_results:
            label = result.get("label", "").lower()
            confidence = float(result.get("score", 0.0))
            for mood, indicators in self.mood_visual_elements.items():
                for indicator in indicators:
                    if indicator in label:
                        mood_scores[mood] = max(mood_scores[mood], confidence * 0.5)

        return mood_scores

    def process_multimodal(
        self,
        text: Optional[str] = None,
        image: Optional[Union[Image.Image, str, bytes]] = None,
        audio: Optional[bytes] = None,
    ) -> MultiModalAnalysis:
        """
        Process multi-modal input and return combined analysis.
        """
        text_analysis = self.process_text(text) if text else {}
        image_analysis = self.process_image(image) if image else {}
        audio_analysis = self.process_audio(audio) if audio else {}

        combined_text = text or ""
        if audio_analysis.get("transcript"):
            combined_text += " " + audio_analysis["transcript"]

        mood_categories = self._extract_mood_categories(combined_text)

        if image_analysis.get("mood_indicators"):
            image_moods = [
                mood
                for mood, score in image_analysis["mood_indicators"].items()
                if score > 0.3
            ]
            mood_categories.extend(image_moods)

        mood_categories = list(set(mood_categories))
        primary_mood = mood_categories[0] if mood_categories else "general"

        text_confidence = 0.8 if text_analysis else 0.0
        image_confidence = float(image_analysis.get("food_confidence", 0.0) or 0.0)
        audio_confidence = 0.7 if audio_analysis.get("transcript") else 0.0

        total_weight = (
            (1.0 if text_analysis else 0.0)
            + (0.8 if image_analysis else 0.0)
            + (0.6 if audio_analysis else 0.0)
        )
        combined_confidence = (
            text_confidence * (1.0 if text_analysis else 0.0)
            + image_confidence * 0.8 * (1.0 if image_analysis else 0.0)
            + audio_confidence * 0.6 * (1.0 if audio_analysis else 0.0)
        ) / total_weight if total_weight > 0 else 0.0

        entities = self._extract_entities(combined_text)

        return MultiModalAnalysis(
            primary_mood=primary_mood,
            confidence=combined_confidence,
            mood_categories=mood_categories,
            extracted_entities=entities,
            image_analysis=image_analysis or None,
            audio_analysis=audio_analysis or None,
            combined_confidence=combined_confidence,
        )

    def _extract_mood_categories(self, text: str) -> List[str]:
        text_lower = text.lower()
        mood_categories: List[str] = []
        mood_keywords = {
            "comfort": ["comfort", "warm", "cozy", "soothing", "nurturing"],
            "excitement": ["exciting", "spicy", "bold", "energetic", "vibrant"],
            "romance": ["romantic", "elegant", "sophisticated", "intimate"],
            "health": ["healthy", "fresh", "light", "clean", "natural"],
            "celebration": ["celebration", "party", "festive", "special"],
            "weather_hot": ["hot", "summer", "refreshing", "cool"],
            "weather_cold": ["cold", "winter", "warm", "hearty"],
            "energy_light": ["light", "easy", "gentle", "subtle"],
            "energy_heavy": ["heavy", "filling", "substantial", "rich"],
        }
        for mood, keywords in mood_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                mood_categories.append(mood)
        return mood_categories

    def _extract_entities(self, text: str) -> List[str]:
        entities: List[str] = []
        text_lower = text.lower()
        entity_patterns = {
            "weather": ["hot", "cold", "warm", "cool", "sunny", "rainy"],
            "time": ["morning", "afternoon", "evening", "night"],
            "social": ["alone", "couple", "family", "friends"],
            "flavor": ["sweet", "spicy", "salty", "savory"],
            "energy": ["light", "heavy", "greasy", "filling"],
        }
        for patterns in entity_patterns.values():
            for pattern in patterns:
                if pattern in text_lower:
                    entities.append(pattern)
        return entities

    def get_processing_info(self) -> Dict[str, Any]:
        return {
            "text_model": self.text_model_name,
            "image_model": self.image_model_name,
            "device": self.device,
            "food_labels_count": len(self.food_labels),
            "mood_elements_count": len(self.mood_visual_elements),
            "models_loaded": {
                "text": self.text_model is not None,
                "image_pipelines": self.image_classifier is not None and self.image_captioner is not None,
                "speech": self.speech_recognizer is not None,
            },
        }


# Convenience functions

def process_text_input(text: str) -> Dict[str, Any]:
    processor = MultiModalProcessor()
    return processor.process_text(text)


def process_image_input(image: Union[Image.Image, str, bytes]) -> Dict[str, Any]:
    processor = MultiModalProcessor()
    return processor.process_image(image)


def process_audio_input(audio_data: bytes) -> Dict[str, Any]:
    processor = MultiModalProcessor()
    return processor.process_audio(audio_data)


def process_combined_input(
    text: Optional[str] = None,
    image: Optional[Union[Image.Image, str, bytes]] = None,
    audio: Optional[bytes] = None,
) -> MultiModalAnalysis:
    processor = MultiModalProcessor()
    return processor.process_multimodal(text=text, image=image, audio=audio) 