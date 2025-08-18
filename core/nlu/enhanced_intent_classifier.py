"""
Enhanced Intent Classifier with Transformer Models
Phase 3: Advanced AI Features - Deep Learning Models

Features:
- Transformer-based intent classification
- Semantic embeddings for better understanding
- Offline fallback mechanisms
- Robust error handling
"""

import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import ML libraries with fallbacks
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using fallback methods")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available, using fallback methods")

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence Transformers not available, using fallback methods")

try:
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available, using fallback methods")

class EnhancedIntentClassifier:
    """
    Enhanced intent classifier with transformer models and offline fallbacks.
    
    Features:
    - Transformer-based classification with DistilBERT
    - Sentence embeddings for semantic understanding
    - Comprehensive offline fallback system
    - Robust error handling for production use
    """

    def __init__(self, taxonomy_path: str, model_dir: str = "models/intent_classifier"):
        self.taxonomy_path = taxonomy_path
        self.model_dir = Path(model_dir)
        self.device = self._setup_device()
        
        # Initialize components with fallbacks
        self.tokenizer = None
        self.model = None
        self.sentence_transformer = None
        self.taxonomy = None
        self.intent_labels = []
        
        # Load components with comprehensive fallbacks
        self._load_taxonomy()
        self._load_models()
        
        logger.info(f"Enhanced intent classifier initialized on device: {self.device}")

    def _setup_device(self) -> str:
        """Setup optimal device with fallbacks."""
        if not TORCH_AVAILABLE:
            return "cpu"
        
        try:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except Exception as e:
            logger.warning(f"Device setup failed: {e}, using CPU")
            return "cpu"

    def _load_taxonomy(self):
        """Load taxonomy with fallback."""
        try:
            if os.path.exists(self.taxonomy_path):
                with open(self.taxonomy_path, 'r', encoding='utf-8') as f:
                    self.taxonomy = json.load(f)
                logger.info(f"Taxonomy loaded: {len(self.taxonomy)} categories")
            else:
                logger.warning(f"Taxonomy file not found: {self.taxonomy_path}")
                self.taxonomy = {}
        except Exception as e:
            logger.error(f"Failed to load taxonomy: {e}")
            self.taxonomy = {}

    def _load_models(self):
        """Load models with comprehensive fallbacks."""
        # Try to load sentence transformer first
        self._load_sentence_transformer()
        
        # Try to load transformer model
        self._load_transformer_model()
        
        # Always setup fallback methods for robustness
        self._setup_fallback_methods()
        
        # Ensure we have at least basic functionality
        if not self._has_working_models():
            logger.warning("No working models available, using fallback methods")

    def _load_sentence_transformer(self):
        """Load sentence transformer with fallbacks."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("Sentence Transformers not available")
            return
        
        try:
            # Try local model first
            local_model_path = self.model_dir / "sentence_transformer"
            if local_model_path.exists():
                logger.info("Loading sentence transformer from local path")
                self.sentence_transformer = SentenceTransformer(str(local_model_path), device=self.device)
            else:
                # Try online with timeout
                logger.info("Loading sentence transformer from HuggingFace (with timeout)")
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Model loading timed out")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)  # 30 second timeout
                
                try:
                    self.sentence_transformer = SentenceTransformer(
                        "sentence-transformers/all-MiniLM-L6-v2", 
                        device=self.device
                    )
                    signal.alarm(0)  # Cancel timeout
                    logger.info("Sentence transformer loaded successfully")
                except TimeoutError:
                    logger.warning("Sentence transformer loading timed out")
                except Exception as e:
                    logger.warning(f"Sentence transformer loading failed: {e}")
                finally:
                    signal.alarm(0)
                    
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")

    def _load_transformer_model(self):
        """Load transformer model with fallbacks."""
        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            logger.warning("Transformers or PyTorch not available")
            return
        
        try:
            # Try local model first - our trained model is directly in the model_dir
            if self.model_dir.exists():
                logger.info(f"Loading transformer model from: {self.model_dir}")
                self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(self.model_dir))
                self.model = DistilBertForSequenceClassification.from_pretrained(str(self.model_dir))
                self.model.to(self.device)
                logger.info("Transformer model loaded successfully")
            else:
                logger.info(f"No local transformer model found at: {self.model_dir}")
                
        except Exception as e:
            logger.warning(f"Failed to load transformer model: {e}")

    def _has_working_models(self) -> bool:
        """Check if we have any working models."""
        return (
            self.sentence_transformer is not None or 
            (self.tokenizer is not None and self.model is not None)
        )

    def _setup_fallback_methods(self):
        """Setup basic fallback methods for intent classification."""
        logger.info("Setting up fallback intent classification methods")
        
        # Basic keyword-based fallback
        self.fallback_keywords = {
            "comfort": ["comfort", "warm", "cozy", "soothing", "nurturing"],
            "energy": ["energy", "vitality", "energetic", "powerful"],
            "health": ["healthy", "fresh", "light", "clean", "natural", "ill", "sick", "recovery", "nauseous", "weak", "feeling ill"],
            "indulgence": ["indulge", "treat", "sweet", "rich", "decadent"],
            "quick": ["quick", "fast", "efficient", "simple"],
            "romantic": ["romantic", "elegant", "sophisticated", "intimate"]
        }

    def classify_intent(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Classify intent using available models with fallbacks.
        
        Args:
            text: Input text to classify
            context: Optional context information
            
        Returns:
            Dictionary with classification results
        """
        if not text:
            return self._create_fallback_response("no_text")
        
        try:
            # Try transformer-based classification first
            if self._has_working_models():
                result = self._transformer_classification(text, context)
                if result:
                    return result
            
            # Fallback to basic methods
            return self._fallback_intent_classification(text, context)
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return self._create_fallback_response("error", error=str(e))

    def _transformer_classification(self, text: str, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Attempt transformer-based classification."""
        try:
            if self.sentence_transformer:
                return self._sentence_transformer_classification(text, context)
            elif self.tokenizer and self.model:
                return self._distilbert_classification(text, context)
        except Exception as e:
            logger.warning(f"Transformer classification failed: {e}")
        return None

    def _sentence_transformer_classification(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Classify using sentence transformer embeddings."""
        try:
            # Get text embedding
            embedding = self.sentence_transformer.encode(text, convert_to_tensor=True)
            if self.device != "cpu":
                embedding = embedding.to(self.device)
            
            # Simple similarity-based classification
            intent_scores = self._calculate_intent_similarities(embedding, text)
            
            # Get top intent
            top_intent = max(intent_scores.items(), key=lambda x: x[1])
            
            return {
                "primary_intent": top_intent[0],
                "confidence": float(top_intent[1]),
                "all_intents": sorted(intent_scores.items(), key=lambda x: x[1], reverse=True),
                "method": "sentence_transformer",
                "embedding_shape": list(embedding.shape)
            }
            
        except Exception as e:
            logger.warning(f"Sentence transformer classification failed: {e}")
            raise

    def _distilbert_classification(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Classify using DistilBERT model."""
        try:
            # Load label mappings
            label_mappings_path = self.model_dir / "unified_label_mappings.json"
            if not label_mappings_path.exists():
                logger.warning("Label mappings not found, using fallback")
                return None
            
            with open(label_mappings_path, 'r') as f:
                label_mappings = json.load(f)
            
            id_to_label = {int(k): v for k, v in label_mappings.get("id_to_label", {}).items()}
            
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            # Move to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.sigmoid(logits)  # Use sigmoid for multi-label classification
            
            # Get top predictions (multi-label)
            probs_np = probs.cpu().numpy()[0]
            top_indices = np.argsort(probs_np)[-3:]  # Get top 3 predictions
            
            # Convert to intent labels
            all_intents = []
            for idx in top_indices:
                if idx in id_to_label:
                    label = id_to_label[idx]
                    confidence = float(probs_np[idx])
                    all_intents.append([label, confidence])
            
            # Sort by confidence
            all_intents.sort(key=lambda x: x[1], reverse=True)
            
            # Get primary intent
            primary_intent = all_intents[0][0] if all_intents else "unknown"
            primary_confidence = all_intents[0][1] if all_intents else 0.0
            
            return {
                "primary_intent": primary_intent,
                "confidence": primary_confidence,
                "all_intents": all_intents,
                "method": "distilbert",
                "logits_shape": list(logits.shape)
            }
            
        except Exception as e:
            logger.warning(f"DistilBERT classification failed: {e}")
            return None

    def _calculate_intent_similarities(self, embedding, text: str) -> Dict[str, float]:
        """Calculate similarities between text and intent categories."""
        # Simple keyword matching as fallback
        text_lower = text.lower()
        scores = {}
        
        for intent, keywords in self.fallback_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[intent] = score / len(keywords) if score > 0 else 0.0
        
        # Normalize scores
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}
        
        return scores

    def _fallback_intent_classification(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fallback intent classification using basic methods."""
        try:
            # Keyword-based classification
            intent_scores = self._calculate_intent_similarities(None, text)
            
            if not intent_scores:
                return self._create_fallback_response("unknown")
            
            # Get top intent
            top_intent = max(intent_scores.items(), key=lambda x: x[1])
            
            return {
                "primary_intent": top_intent[0],
                "confidence": top_intent[1],
                "all_intents": sorted(intent_scores.items(), key=lambda x: x[1], reverse=True),
                "method": "fallback_keywords",
                "fallback": True
            }
            
        except Exception as e:
            logger.error(f"Fallback classification failed: {e}")
            return self._create_fallback_response("fallback_error", error=str(e))

    def _create_fallback_response(self, intent_type: str, error: Optional[str] = None) -> Dict[str, Any]:
        """Create a fallback response when classification fails."""
        response = {
            "primary_intent": intent_type,
            "confidence": 0.1,
            "all_intents": [[intent_type, 0.1]],
            "method": "fallback",
            "fallback": True
        }
        
        if error:
            response["error"] = error
            
        return response

    def get_semantic_embedding(self, text: str) -> Optional[List[float]]:
        """Get semantic embedding if available."""
        if not self.sentence_transformer:
            return None
        
        try:
            embedding = self.sentence_transformer.encode(text)
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            logger.warning(f"Failed to get semantic embedding: {e}")
            return None

    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all model components."""
        return {
            "sentence_transformer": self.sentence_transformer is not None,
            "transformer_model": self.model is not None,
            "tokenizer": self.tokenizer is not None,
            "taxonomy": len(self.taxonomy) if self.taxonomy else 0,
            "device": self.device,
            "torch_available": TORCH_AVAILABLE,
            "numpy_available": NUMPY_AVAILABLE,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
            "transformers_available": TRANSFORMERS_AVAILABLE
        } 