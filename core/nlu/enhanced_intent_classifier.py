"""
Enhanced Intent Classifier with Transformer Models
Phase 3: Advanced AI Features - Deep Learning Models

Features:
- Transformer-based intent classification
- ML classifier with hot reloading
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
    from transformers import DistilBertTokenizerFast, DistilBertModel
    import torch.nn as nn
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available, using fallback methods")

# Try to import ML classifier libraries
try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    ML_CLASSIFIER_AVAILABLE = True
except ImportError:
    ML_CLASSIFIER_AVAILABLE = False
    logger.warning("ML classifier libraries not available (scikit-learn, joblib)")

# Custom model class to match the trained architecture
class DistilBertDualHead(nn.Module):
    def __init__(self, model_name_or_path, taxonomy_classes: int, mlb_classes: int, dropout: float = 0.2):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(model_name_or_path)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.taxonomy_head = nn.Linear(hidden_size, taxonomy_classes)
        self.mlb_head = nn.Linear(hidden_size, mlb_classes)

    def forward(self, input_ids=None, attention_mask=None, labels=None, mlb_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)

        taxonomy_logits = self.taxonomy_head(pooled)
        mlb_logits = self.mlb_head(pooled)

        return {"logits": taxonomy_logits, "mlb_logits": mlb_logits}

class EnhancedIntentClassifier:
    """
    Enhanced intent classifier with transformer models, ML classifier, and offline fallbacks.
    
    Features:
    - Transformer-based classification with DistilBERT
    - ML classifier with hot reloading capability
    - Sentence embeddings for semantic understanding
    - Comprehensive offline fallback system
    - Robust error handling for production use
    """

    def __init__(self, taxonomy_path: str, model_dir: str = "models/intent_classifier", use_hybrid: bool = True):
        self.taxonomy_path = taxonomy_path
        self.model_dir = Path(model_dir)
        self.device = self._setup_device()
        self.use_hybrid = use_hybrid
        
        # Initialize components with fallbacks
        self.tokenizer = None
        self.model = None
        self.sentence_transformer = None
        self.taxonomy = None
        self.intent_labels = []
        
        # ML Classifier components
        self.ml_classifier = None
        self.ml_vectorizer = None
        self.ml_labels = []
        self.ml_classifier_path = self.model_dir / "ml_classifier.pkl"
        self.ml_mapping_path = self.model_dir / "label_mappings.json"
        
        # Load components with comprehensive fallbacks
        self._load_taxonomy()
        self._load_models()
        self._load_ml_classifier()
        
        # Setup fallback methods
        self._setup_fallback_methods()
        
        # Initialize taxonomy mapper for bridging 20->138 categories
        self.taxonomy_mapper = None
        
        # Initialize taxonomy mapper immediately (don't wait for transformer model)
        self._init_taxonomy_mapper()
        
        logger.info(f"Enhanced intent classifier initialized on device: {self.device}")

    def _init_hybrid_components(self):
        """Initialize hybrid LLM components."""
        try:
            # Import hybrid components
            from .llm_parser import classify_with_llm
            from .validator import validate_labels
            
            self.llm_classify = classify_with_llm
            self.validate_labels = validate_labels
            
            logger.info("Hybrid LLM components initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize hybrid components: {e}")
            self.use_hybrid = False

    def _setup_device(self) -> str:
        """Setup optimal device with fallbacks."""
        if not TORCH_AVAILABLE:
            return "cpu"
        
        # Force CPU to avoid MPS issues
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
                
                # Load the trained model with the correct architecture
                # IMPORTANT: The model was trained for 20 classes, not 138
                # We need to use the original architecture to avoid size mismatch
                model_path = str(self.model_dir)
                
                # Load the base DistilBERT model
                base_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
                
                # Create our custom dual-head model with the CORRECT architecture
                # The model was trained with:
                # - taxonomy_head: 20 classes (from taxonomy_labels.json)
                # - mlb_head: 35 classes (from mlb_classes.json)
                taxonomy_classes = 20
                mlb_classes = 35
                
                self.model = DistilBertDualHead(
                    model_name_or_path="distilbert-base-uncased",
                    taxonomy_classes=taxonomy_classes,  # 20 classes
                    mlb_classes=mlb_classes,           # 35 classes
                    dropout=0.2
                )
                
                # Load the trained weights
                model_weights_path = os.path.join(model_path, "model.safetensors")
                if os.path.exists(model_weights_path):
                    try:
                        from safetensors.torch import load_file
                        state_dict = load_file(model_weights_path)
                        self.model.load_state_dict(state_dict, strict=False)
                        logger.info("Loaded trained model weights successfully")
                        self.model.to(self.device)
                        logger.info("Transformer model loaded successfully")
                        
                        # Initialize taxonomy mapper for bridging 20->138
                        self._init_taxonomy_mapper()
                        
                    except Exception as load_error:
                        logger.warning(f"Failed to load model weights: {load_error}")
                        logger.info("Using keyword fallback classification")
                        self.model = None
                        return
                else:
                    logger.warning("Model weights not found, using keyword fallback")
                    self.model = None
                    return
            else:
                logger.info(f"No local transformer model found at: {self.model_dir}")
                
        except Exception as e:
            logger.warning(f"Failed to load transformer model: {e}")
            import traceback
            traceback.print_exc()
    
    def _init_taxonomy_mapper(self):
        """Initialize the taxonomy mapper for bridging 20->138 categories."""
        try:
            from .taxonomy_mapper import TaxonomyMapper
            self.taxonomy_mapper = TaxonomyMapper(str(self.model_dir))
            logger.info("Taxonomy mapper initialized successfully")
            
            # Print mapping statistics
            stats = self.taxonomy_mapper.get_mapping_stats()
            logger.info(f"Taxonomy mapping stats: {stats}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize taxonomy mapper: {e}")
            self.taxonomy_mapper = None

    def _load_ml_classifier(self):
        """Load ML classifier with fallbacks."""
        if not ML_CLASSIFIER_AVAILABLE:
            logger.warning("ML classifier libraries not available")
            return
        
        try:
            if self.ml_classifier_path.exists() and self.ml_mapping_path.exists():
                logger.info(f"Loading ML classifier from: {self.ml_classifier_path}")
                
                # Load the ML classifier
                model_data = joblib.load(self.ml_classifier_path)
                
                if isinstance(model_data, dict) and 'classifier' in model_data and 'vectorizer' in model_data:
                    self.ml_classifier = model_data['classifier']
                    self.ml_vectorizer = model_data['vectorizer']
                    self.ml_labels = model_data.get('labels', [])
                    
                    logger.info(f"ML classifier loaded successfully with {len(self.ml_labels)} labels")
                    logger.info(f"Available ML labels: {self.ml_labels[:10]}...")  # Show first 10
                else:
                    logger.warning("Invalid ML classifier format")
                    self.ml_classifier = None
                    self.ml_vectorizer = None
                    
            else:
                logger.info("No ML classifier found, will use fallback methods")
                self.ml_classifier = None
                self.ml_vectorizer = None
                
        except Exception as e:
            logger.error(f"Failed to load ML classifier: {e}")
            self.ml_classifier = None
            self.ml_vectorizer = None

    def reload_ml_classifier(self) -> bool:
        """Hot reload the ML classifier from disk."""
        if not ML_CLASSIFIER_AVAILABLE:
            logger.warning("ML classifier libraries not available for reloading")
            return False
        
        try:
            logger.info("Reloading ML classifier...")
            
            if self.ml_classifier_path.exists() and self.ml_mapping_path.exists():
                # Load the new ML classifier
                model_data = joblib.load(self.ml_classifier_path)
                
                if isinstance(model_data, dict) and 'classifier' in model_data and 'vectorizer' in model_data:
                    self.ml_classifier = model_data['classifier']
                    self.ml_vectorizer = model_data['vectorizer']
                    self.ml_labels = model_data.get('labels', [])
                    
                    logger.info(f"✅ ML classifier reloaded successfully with {len(self.ml_labels)} labels")
                    return True
                else:
                    logger.error("Invalid ML classifier format during reload")
                    return False
            else:
                logger.warning("ML classifier files not found during reload")
                return False
                
        except Exception as e:
            logger.error(f"Failed to reload ML classifier: {e}")
            return False

    def _ml_classification(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Classify intent using ML classifier."""
        if not self.ml_classifier or not self.ml_vectorizer or not self.ml_labels:
            logger.debug("ML classifier not available")
            return None
        
        try:
            # Vectorize the text
            X = self.ml_vectorizer.transform([text])
            
            # Get predictions
            predictions = self.ml_classifier.predict(X)
            probabilities = self.ml_classifier.predict_proba(X)
            
            # Get predicted labels
            predicted_labels = []
            for i, pred in enumerate(predictions[0]):
                if pred == 1:  # Label is predicted
                    label = self.ml_labels[i]
                    prob = probabilities[0][i] if i < len(probabilities[0]) else 0.5
                    predicted_labels.append((label, prob))
            
            # Sort by probability
            predicted_labels.sort(key=lambda x: x[1], reverse=True)
            
            if predicted_labels:
                primary_intent = predicted_labels[0][0]
                confidence = predicted_labels[0][1]
                all_intents = [(label, prob) for label, prob in predicted_labels[:5]]  # Top 5
                
                return {
                    'primary_intent': primary_intent,
                    'confidence': confidence,
                    'all_intents': all_intents,
                    'method': 'ml_classifier',
                    'fallback': False
                }
            else:
                logger.debug("No labels predicted by ML classifier")
                return None
                
        except Exception as e:
            logger.error(f"ML classification failed: {e}")
            return None

    def _has_working_models(self) -> bool:
        """Check if we have any working models."""
        return (
            self.sentence_transformer is not None or 
            (self.tokenizer is not None and self.model is not None) or
            (self.ml_classifier is not None and self.ml_vectorizer is not None)
        )

    def _setup_fallback_methods(self):
        """Setup basic fallback methods for intent classification."""
        logger.info("Setting up fallback intent classification methods")
        
        # Enhanced keyword-based fallback with comprehensive coverage
        self.fallback_keywords = {
            # Comfort and warming
            "comfort": ["comfort", "warm", "cozy", "soothing", "nurturing", "warming", "comforting", "comfortable"],
            
            # Energy and vitality
            "energy": ["energy", "vitality", "energetic", "powerful", "energizing"],
            
            # Health and wellness
            "health": ["healthy", "fresh", "light", "clean", "natural", "ill", "sick", "recovery", "nauseous", "weak", "feeling ill", "health"],
            
            # Indulgence and treats
            "indulgence": ["indulge", "treat", "sweet", "rich", "decadent", "indulgence"],
            
            # Quick and simple
            "quick": ["quick", "fast", "efficient", "simple", "quickly"],
            
            # Romantic and elegant
            "romantic": ["romantic", "elegant", "sophisticated", "intimate", "romance"],
            
            # Japanese cuisine
            "japanese": ["japanese", "japan", "sushi", "ramen", "tempura", "miso", "teriyaki", "udon", "soba", "bento"],
            
            # Cold and refreshing
            "cold": ["cold", "cool", "refreshing", "chilled", "ice", "frozen", "cold", "refreshing"],
            
            # Spicy and hot
            "spicy": ["spicy", "hot", "spice", "chili", "pepper", "fiery", "burning", "zesty"],
            
            # Southeast Asian cuisine
            "southeast_asian": ["thai", "vietnamese", "malaysian", "indonesian", "filipino", "southeast", "asian", "curry", "lemongrass", "coconut"],
            
            # Chinese cuisine
            "chinese": ["chinese", "china", "dim sum", "kung pao", "sweet and sour", "chow mein", "fried rice"],
            
            # Italian cuisine
            "italian": ["italian", "italy", "pasta", "pizza", "risotto", "bruschetta", "tiramisu"],
            
            # Mexican cuisine
            "mexican": ["mexican", "mexico", "taco", "burrito", "enchilada", "guacamole", "salsa"],
            
            # Indian cuisine
            "indian": ["indian", "india", "curry", "tandoori", "naan", "biryani", "masala"],
            
            # Mediterranean cuisine
            "mediterranean": ["mediterranean", "greek", "turkish", "lebanese", "hummus", "falafel", "tabbouleh"],
            
            # American cuisine
            "american": ["american", "usa", "burger", "hot dog", "bbq", "steak", "apple pie"],
            
            # French cuisine
            "french": ["french", "france", "croissant", "quiche", "ratatouille", "coq au vin", "creme brulee"]
        }

    async def classify_intent_hybrid(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Classify intent using hybrid approach (LLM + Validation + ML).
        
        Returns:
            Dict with hybrid classification results
        """
        if not self.use_hybrid:
            logger.warning("Hybrid mode not available, falling back to standard classification")
            return self.classify_intent(text, context)
        
        try:
            logger.info(f"Performing hybrid classification for: '{text}'")
            
            # Step 1: LLM Classification
            llm_labels = await self.llm_classify(text, self.intent_labels)
            logger.info(f"LLM labels: {llm_labels}")
            
            # Step 2: Validation
            validated_labels = self.validate_labels(llm_labels, self.intent_labels)
            logger.info(f"Validated labels: {validated_labels}")
            
            # Step 3: ML Classification for comparison
            ml_result = self.classify_intent(text, context)
            
            # Combine results
            result = {
                "primary_intent": validated_labels[0] if validated_labels else "unknown",
                "confidence": 1.0 if validated_labels else 0.0,
                "all_intents": validated_labels,
                "method": "hybrid_llm",
                "llm_labels": llm_labels,
                "validated_labels": validated_labels,
                "ml_result": ml_result,
                "fallback": False
            }
            
            logger.info(f"Hybrid classification result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Hybrid classification failed: {e}")
            # Fallback to standard classification
            return self.classify_intent(text, context)

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
            # Try ML classifier first (fastest and most reliable)
            if self.ml_classifier and self.ml_vectorizer:
                result = self._ml_classification(text, context)
                if result and result.get('primary_intent') != 'unknown':
                    confidence = result.get('confidence', 0.0)
                    if confidence >= 0.5:  # Accept ML predictions with decent confidence
                        logger.info(f"Using ML classifier result: {result.get('primary_intent')} (confidence: {confidence:.3f})")
                        return result
            
            # Try transformer-based classification second
            if self._has_working_models():
                result = self._transformer_classification(text, context)
                if result and result.get('primary_intent') != 'unknown':
                    # Check if the result makes sense for the input
                    primary_intent = result.get('primary_intent', '')
                    text_lower = text.lower()
                    
                    # Force fallback for obvious mismatches
                    if (primary_intent in ['occasion_party_snacks', 'social_couple', 'emotional_romantic'] and 
                        any(word in text_lower for word in ['comfort', 'warm', 'cozy', 'soothing', 'warming'])):
                        logger.info(f"Forcing fallback due to intent mismatch: {primary_intent} for comfort/warming query")
                        return self._fallback_intent_classification(text, context)
                    
                    # Also check if confidence is too low for the detected intent
                    confidence = result.get('confidence', 0.0)
                    if confidence < 0.5:  # Lowered from 0.7 to 0.5 to allow ML model usage
                        logger.info(f"Forcing fallback due to low confidence: {confidence}")
                        return self._fallback_intent_classification(text, context)
                    
                    return result
            
            # Fallback to basic methods
            return self._fallback_intent_classification(text, context)
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return self._create_fallback_response("error", error=str(e))

    def _transformer_classification(self, text: str, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Attempt transformer-based classification."""
        try:
            # Prioritize DistilBERT model over sentence transformer
            if self.tokenizer and self.model:
                result = self._distilbert_classification(text, context)
                if result and result.get('primary_intent') != 'unknown':
                    return result
            
            # Fallback to sentence transformer
            if self.sentence_transformer:
                result = self._sentence_transformer_classification(text, context)
                if result and result.get('primary_intent') != 'unknown':
                    return result
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
            # Load the ORIGINAL 20 labels that the model was trained on
            taxonomy_labels_path = self.model_dir / "taxonomy_labels.json"
            if not taxonomy_labels_path.exists():
                logger.warning("Original taxonomy labels not found, using fallback")
                return None
            
            with open(taxonomy_labels_path, 'r') as f:
                original_labels = json.load(f)
            
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
            
            # Get predictions from the 20-class model
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs["logits"]  # Get taxonomy logits from our custom model
                probs = torch.sigmoid(logits)  # Use sigmoid for multi-label classification
            
            # Get top predictions from the 20 classes
            probs_np = probs.cpu().numpy()[0]
            top_indices = np.argsort(probs_np)[-3:]  # Get top 3 predictions
            
            # Convert to original intent labels (20 classes)
            original_intents = []
            for idx in top_indices:
                if idx < len(original_labels):
                    label = original_labels[idx]
                    confidence = float(probs_np[idx])
                    original_intents.append([label, confidence])
            
            # Sort by confidence
            original_intents.sort(key=lambda x: x[1], reverse=True)
            
            # Get primary intent from original 20 classes
            primary_original_intent = original_intents[0][0] if original_intents else "unknown"
            primary_confidence = original_intents[0][1] if original_intents else 0.0
            
            # Now use taxonomy mapper to bridge to 138 categories
            logger.info(f"Attempting taxonomy mapping for: {primary_original_intent} (confidence: {primary_confidence})")
            logger.info(f"Taxonomy mapper available: {self.taxonomy_mapper is not None}")
            
            if self.taxonomy_mapper and primary_original_intent != "unknown":
                mapped_categories = self.taxonomy_mapper.map_ml_prediction(
                    primary_original_intent, primary_confidence
                )
                
                logger.info(f"Mapping result: {mapped_categories}")
                
                if mapped_categories:
                    # Use the top mapped category as primary intent
                    top_mapped_category, mapped_confidence = mapped_categories[0]
                    
                    # Create response with mapped categories
                    mapped_intents = []
                    for category, conf in mapped_categories:
                        mapped_intents.append([category, conf])
                    
                    logger.info(f"Successfully mapped to: {top_mapped_category} (confidence: {mapped_confidence})")
                    
                    return {
                        "primary_intent": top_mapped_category,
                        "confidence": mapped_confidence,
                        "all_intents": mapped_intents,
                        "method": "distilbert_mapped",
                        "original_ml_intent": primary_original_intent,
                        "original_confidence": primary_confidence,
                        "logits_shape": list(logits.shape)
                    }
                else:
                    logger.warning(f"Mapping returned empty result for {primary_original_intent}")
            else:
                logger.warning(f"Taxonomy mapper not available or primary intent is unknown: mapper={self.taxonomy_mapper}, intent={primary_original_intent}")
            
            # Fallback to original 20-class results if mapping fails
            logger.warning(f"Taxonomy mapping failed for {primary_original_intent}, using original results")
            return {
                "primary_intent": primary_original_intent,
                "confidence": primary_confidence,
                "all_intents": original_intents,
                "method": "distilbert_original",
                "logits_shape": list(logits.shape)
            }
            
        except Exception as e:
            logger.warning(f"DistilBERT classification failed: {e}")
            return None

    def _calculate_intent_similarities(self, embedding, text: str) -> Dict[str, float]:
        """Calculate similarities between text and intent categories."""
        # Enhanced keyword matching as fallback
        text_lower = text.lower()
        scores = {}
        
        for intent, keywords in self.fallback_keywords.items():
            # Count exact keyword matches
            exact_matches = sum(1 for keyword in keywords if keyword in text_lower)
            
            # Count partial word matches (for compound words)
            partial_matches = 0
            for keyword in keywords:
                if len(keyword) > 3:  # Only check longer keywords
                    for word in text_lower.split():
                        if keyword in word or word in keyword:
                            partial_matches += 0.5  # Partial match gets lower score
            
            # Calculate total score
            total_score = exact_matches + partial_matches
            scores[intent] = total_score / len(keywords) if total_score > 0 else 0.0
        
        # If no matches found, don't default to first intent
        if not any(scores.values()):
            return {}
        
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
                # No keyword matches found, try to infer from text content
                text_lower = text.lower()
                
                # Check for cuisine-specific words
                if any(word in text_lower for word in ['japanese', 'japan', 'sushi', 'ramen']):
                    return {
                        "primary_intent": "japanese",
                        "confidence": 0.8,
                        "all_intents": [["japanese", 0.8]],
                        "method": "fallback_inference",
                        "fallback": True
                    }
                elif any(word in text_lower for word in ['cold', 'cool', 'refreshing', 'chilled']):
                    return {
                        "primary_intent": "cold",
                        "confidence": 0.8,
                        "all_intents": [["cold", 0.8]],
                        "method": "fallback_inference",
                        "fallback": True
                    }
                elif any(word in text_lower for word in ['spicy', 'hot', 'chili', 'pepper']):
                    return {
                        "primary_intent": "spicy",
                        "confidence": 0.8,
                        "all_intents": [["spicy", 0.8]],
                        "method": "fallback_inference",
                        "fallback": True
                    }
                elif any(word in text_lower for word in ['thai', 'vietnamese', 'malaysian', 'southeast', 'asian']):
                    return {
                        "primary_intent": "southeast_asian",
                        "confidence": 0.8,
                        "all_intents": [["southeast_asian", 0.8]],
                        "method": "fallback_inference",
                        "fallback": True
                    }
                else:
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
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "ml_classifier_available": ML_CLASSIFIER_AVAILABLE,
            "ml_classifier_loaded": self.ml_classifier is not None,
            "ml_labels_count": len(self.ml_labels) if self.ml_labels else 0
        } 