"""
Multi-Modal Input Processor
Phase 3: Advanced AI Features - Multi-modal Input Support

Supports:
- Text input (natural language)
- Image input (food photos, mood indicators)
- Voice input (speech-to-text)
- Combined multi-modal analysis
- Comprehensive offline fallbacks
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
import signal
import time

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
    - Text processing with transformers (lazy init with timeouts)
    - Image analysis (food recognition, mood indicators) (lazy init with timeouts)
    - Speech-to-text processing (if available) (lazy init with timeouts)
    - Multi-modal fusion for enhanced understanding
    - Comprehensive offline fallbacks
    """

    def __init__(
        self,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        image_model_name: str = "microsoft/resnet-50",
        device: Optional[str] = None,
        timeout_seconds: int = 30,
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
        self.timeout_seconds = timeout_seconds

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
            "food", "meal", "dish", "cuisine", "restaurant", "cooking", "kitchen",
            "breakfast", "lunch", "dinner", "snack", "dessert", "beverage",
            "fruit", "vegetable", "meat", "fish", "bread", "pasta", "rice"
        ]

        # Mood visual elements
        self.mood_visual_elements = {
            "comfort": ["warm", "cozy", "soft", "gentle", "nurturing"],
            "excitement": ["bright", "vibrant", "bold", "energetic", "dynamic"],
            "romance": ["elegant", "sophisticated", "intimate", "delicate", "refined"],
            "health": ["fresh", "natural", "clean", "light", "organic"],
            "celebration": ["festive", "colorful", "joyful", "sparkling", "cheerful"]
        }

        # Initialize components with fallbacks
        self._initialize_components()

    def _initialize_components(self):
        """Initialize components with comprehensive fallbacks."""
        logger.info("Initializing multi-modal processor components...")
        
        # Initialize text models
        self._ensure_text_models()
        
        # Initialize image models
        self._ensure_image_models()
        
        # Initialize speech models
        self._ensure_speech_models()
        
        logger.info("Multi-modal processor initialization complete")

    def _ensure_text_models(self):
        if self.text_tokenizer is None or self.text_model is None:
            logger.info("Initializing text models for multi-modal processor (lazy)...")
            try:
                # Try local models first
                local_text_path = Path("models") / "text_models"
                if local_text_path.exists():
                    logger.info("Loading text models from local path")
                    self.text_tokenizer = AutoTokenizer.from_pretrained(str(local_text_path / "tokenizer"))
                    self.text_model = AutoModel.from_pretrained(str(local_text_path / "model"))
                else:
                    # Try online with timeout
                    logger.info("Loading text models from HuggingFace (with timeout)")
                    self._load_with_timeout(self._load_text_models_online, "text models")
                    
            except Exception as e:
                logger.warning(f"Text models initialization failed: {e}")
                self.text_tokenizer = None
                self.text_model = None

    def _load_text_models_online(self):
        """Load text models from HuggingFace."""
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
                
                # Try local models first
                local_image_path = Path("models") / "image_models"
                if local_image_path.exists():
                    logger.info("Loading image models from local path")
                    self._load_local_image_models(local_image_path, device_id)
                else:
                    # Try online with timeout
                    logger.info("Loading image models from HuggingFace (with timeout)")
                    self._load_with_timeout(
                        lambda: self._load_online_image_models(device_id), 
                        "image models"
                    )
                    
            except Exception as e:
                logger.error(f"Error initializing image models: {e}")
                # Fallback to basic image processing
                self.image_classifier = None
                self.image_captioner = None

    def _load_local_image_models(self, local_path: Path, device_id: int):
        """Load image models from local storage."""
        try:
            # Load image classifier
            classifier_path = local_path / "classifier"
            if classifier_path.exists():
                self.image_classifier = pipeline(
                    "image-classification",
                    model=str(classifier_path),
                    device=device_id,
                )
                logger.info("Local image classifier loaded")
            
            # Load image captioner
            captioner_path = local_path / "captioner"
            if captioner_path.exists():
                self.image_captioner = pipeline(
                    "image-to-text",
                    model=str(captioner_path),
                    device=device_id,
                )
                logger.info("Local image captioner loaded")
                
        except Exception as e:
            logger.warning(f"Failed to load local image models: {e}")

    def _load_online_image_models(self, device_id: int):
        """Load image models from HuggingFace with individual error handling."""
        # Wrap pipeline creation in try-catch to handle type conversion issues
        try:
            self.image_classifier = pipeline(
                "image-classification",
                model=self.image_model_name,
                device=device_id,
            )
            logger.info("Online image classifier loaded")
        except Exception as e:
            logger.warning(f"Image classifier failed: {e}")
            self.image_classifier = None
        
        try:
            self.image_captioner = pipeline(
                "image-to-text",
                model="nlpconnect/vit-gpt2-image-captioning",
                device=device_id,
            )
            logger.info("Online image captioner loaded")
        except Exception as e:
            logger.warning(f"Image captioner failed: {e}")
            self.image_captioner = None
        
        if self.image_classifier or self.image_captioner:
            logger.info("Image pipelines ready (partial)")
        else:
            logger.warning("All image pipelines failed, using fallback")

    def _ensure_speech_models(self):
        if self.speech_recognizer is None:
            if sr is None:
                logger.warning("SpeechRecognition not available; skipping audio processing")
            else:
                logger.info("Initializing speech recognizer (lazy)...")
                try:
                    self.speech_recognizer = sr.Recognizer()
                    logger.info("Speech recognizer initialized")
                except Exception as e:
                    logger.warning(f"Speech recognizer initialization failed: {e}")

    def _load_with_timeout(self, load_function, component_name: str):
        """Load a component with timeout protection."""
        def timeout_handler(signum, frame):
            raise TimeoutError(f"{component_name} loading timed out")
        
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout_seconds)
        
        try:
            load_function()
            signal.alarm(0)  # Cancel timeout
            logger.info(f"{component_name} loaded successfully")
        except TimeoutError:
            logger.warning(f"{component_name} loading timed out after {self.timeout_seconds}s")
        except Exception as e:
            logger.warning(f"{component_name} loading failed: {e}")
        finally:
            signal.alarm(0)  # Ensure timeout is cancelled

    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process text input using transformer models.
        """
        if not text:
            return {}

        try:
            self._ensure_text_models()

            if self.text_tokenizer is None or self.text_model is None:
                logger.warning("Text models not available, using fallback")
                return self._fallback_text_processing(text)

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

        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            return self._fallback_text_processing(text)

    def _fallback_text_processing(self, text: str) -> Dict[str, Any]:
        """Fallback text processing when models fail."""
        return {
            "text": text,
            "embeddings": None,
            "length": len(text),
            "processed": False,
            "fallback": True,
            "method": "keyword_analysis"
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

            # Handle cases where models failed to initialize
            if self.image_classifier is None and self.image_captioner is None:
                logger.warning("No image models available, using fallback analysis")
                return {
                    "caption": "Image analysis not available",
                    "classification": [],
                    "food_confidence": 0.5,
                    "mood_indicators": ["general"],
                    "processed": False,
                    "fallback": True,
                }

            # Process with available models
            classification_results = []
            caption = ""
            
            if self.image_classifier:
                try:
                    classification_results = self.image_classifier(image)
                except Exception as e:
                    logger.warning(f"Image classification failed: {e}")
                    classification_results = []
            
            if self.image_captioner:
                try:
                    caption_results = self.image_captioner(image)
                    caption = caption_results[0]["generated_text"] if caption_results else ""
                except Exception as e:
                    logger.warning(f"Image captioning failed: {e}")
                    caption = ""

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
            return self._fallback_image_processing(image)

    def _fallback_image_processing(self, image: Image.Image) -> Dict[str, Any]:
        """Fallback image processing when models fail."""
        # Basic image analysis
        try:
            # Get basic image properties
            width, height = image.size
            mode = image.mode
            
            # Simple food detection based on image properties
            food_confidence = 0.3  # Default low confidence
            
            # Adjust confidence based on image characteristics
            if mode == "RGB":
                food_confidence += 0.1
            if width > 200 and height > 200:
                food_confidence += 0.1
            
            return {
                "caption": "Basic image analysis (fallback)",
                "classification": [],
                "food_confidence": food_confidence,
                "mood_indicators": ["general"],
                "processed": False,
                "fallback": True,
                "image_properties": {
                    "width": width,
                    "height": height,
                    "mode": mode
                }
            }
        except Exception as e:
            logger.error(f"Fallback image processing failed: {e}")
            return {
                "caption": "Image processing failed",
                "classification": [],
                "food_confidence": 0.0,
                "mood_indicators": [],
                "processed": False,
                "fallback": True,
                "error": str(e)
            }

    def process_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Process audio input for speech-to-text conversion.
        """
        if not audio_data:
            return {}

        try:
            self._ensure_speech_models()

            if self.speech_recognizer is None:
                logger.warning("Speech recognizer not available")
                return {
                    "transcript": None,
                    "confidence": 0.0,
                    "processed": False,
                    "fallback": True,
                }

            # Convert audio to audio file format
            audio_file = io.BytesIO(audio_data)
            
            # Process with timeout
            try:
                with sr.AudioFile(audio_file) as source:
                    audio = self.speech_recognizer.record(source)
                    transcript = self.speech_recognizer.recognize_google(audio)
                    
                    return {
                        "transcript": transcript,
                        "confidence": 0.8,  # Google doesn't provide confidence
                        "processed": True,
                    }
            except sr.UnknownValueError:
                return {
                    "transcript": None,
                    "confidence": 0.0,
                    "processed": False,
                    "error": "Speech not recognized"
                }
            except sr.RequestError as e:
                return {
                    "transcript": None,
                    "confidence": 0.0,
                    "processed": False,
                    "error": f"Speech recognition service error: {e}"
                }

        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return {
                "transcript": None,
                "confidence": 0.0,
                "processed": False,
                "fallback": True,
                "error": str(e)
            }

    def process_multimodal(
        self,
        text: Optional[str] = None,
        image: Optional[Union[Image.Image, str, bytes]] = None,
        audio: Optional[bytes] = None,
    ) -> MultiModalAnalysis:
        """
        Process multi-modal input and combine results.
        """
        # Process each modality
        text_analysis = self.process_text(text) if text else None
        image_analysis = self.process_image(image) if image else None
        audio_analysis = self.process_audio(audio) if audio else None

        # Combine text from all sources
        combined_text = text or ""
        if image_analysis and image_analysis.get("caption"):
            combined_text += f" {image_analysis['caption']}"
        if audio_analysis and audio_analysis.get("transcript"):
            combined_text += f" {audio_analysis['transcript']}"

        # Analyze combined text for mood
        primary_mood = self._analyze_primary_mood(combined_text)
        mood_categories = self._extract_mood_categories(combined_text)

        # Calculate confidence scores
        text_confidence = 0.8 if text_analysis and text_analysis.get("processed") else 0.0
        image_confidence = float(image_analysis.get("food_confidence", 0.0) or 0.0) if image_analysis else 0.0
        audio_confidence = 0.7 if audio_analysis and audio_analysis.get("transcript") else 0.0

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

    def _analyze_primary_mood(self, text: str) -> str:
        """Analyze text to determine primary mood."""
        text_lower = text.lower()
        
        # Simple keyword-based mood detection
        mood_keywords = {
            "comfort": ["comfort", "warm", "cozy", "soothing", "nurturing"],
            "excitement": ["exciting", "spicy", "bold", "energetic", "vibrant"],
            "romance": ["romantic", "elegant", "sophisticated", "intimate"],
            "health": ["healthy", "fresh", "light", "clean", "natural"],
            "celebration": ["celebration", "party", "festive", "special"],
        }
        
        for mood, keywords in mood_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return mood
        
        return "neutral"

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

    def _analyze_food_content(self, classification_results: List[Dict[str, Any]]) -> float:
        """Analyze classification results for food content."""
        if not classification_results:
            return 0.0
        
        food_score = 0.0
        total_score = 0.0
        
        for result in classification_results:
            label = result.get("label", "").lower()
            score = result.get("score", 0.0)
            
            # Check if label contains food-related keywords
            if any(food_word in label for food_word in self.food_labels):
                food_score += score
            
            total_score += score
        
        return food_score / total_score if total_score > 0 else 0.0

    def _analyze_mood_indicators(self, caption: str, classification_results: List[Dict[str, Any]]) -> List[str]:
        """Analyze caption and classification for mood indicators."""
        mood_indicators = []
        
        # Analyze caption
        if caption:
            caption_lower = caption.lower()
            for mood, elements in self.mood_visual_elements.items():
                if any(element in caption_lower for element in elements):
                    mood_indicators.append(mood)
        
        # Analyze classification results
        for result in classification_results:
            label = result.get("label", "").lower()
            for mood, elements in self.mood_visual_elements.items():
                if any(element in label for element in elements):
                    if mood not in mood_indicators:
                        mood_indicators.append(mood)
        
        return mood_indicators if mood_indicators else ["general"]

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
            "fallback_available": True,
            "timeout_seconds": self.timeout_seconds,
        }

    def get_component_status(self) -> Dict[str, Any]:
        """Get detailed status of all components."""
        return {
            "text_models": {
                "tokenizer": self.text_tokenizer is not None,
                "model": self.text_model is not None,
                "available": self.text_tokenizer is not None and self.text_model is not None
            },
            "image_models": {
                "classifier": self.image_classifier is not None,
                "captioner": self.image_captioner is not None,
                "available": self.image_classifier is not None or self.image_captioner is not None
            },
            "speech_models": {
                "recognizer": self.speech_recognizer is not None,
                "available": self.speech_recognizer is not None
            },
            "device": self.device,
            "fallback_system": True,
            "timeout_protection": True
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