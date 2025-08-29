"""
Taxonomy Mapper: Bridges 20-class ML model with 138-category taxonomy

This module provides intelligent mapping between the original 20 classes
that the ML model was trained on and the new 138 categories in the
extended taxonomy.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class TaxonomyMapper:
    """
    Maps between the original 20 ML classes and the new 138 taxonomy categories.
    
    The ML model was trained on 20 specific categories, but we now have 138.
    This mapper provides intelligent bridging to maintain ML functionality
    while leveraging the expanded taxonomy.
    """
    
    def __init__(self, model_dir: str = "models/intent_classifier"):
        self.model_dir = Path(model_dir)
        self.original_labels = []
        self.unified_mappings = {}
        self.taxonomy_categories = []
        self.mapping_matrix = {}
        
        self._load_mappings()
        self._build_mapping_matrix()
    
    def _load_mappings(self):
        """Load the original labels and unified mappings."""
        try:
            # Load original 20 labels
            taxonomy_labels_path = self.model_dir / "taxonomy_labels.json"
            if taxonomy_labels_path.exists():
                with open(taxonomy_labels_path, 'r', encoding='utf-8') as f:
                    self.original_labels = json.load(f)
                logger.info(f"Loaded {len(self.original_labels)} original ML labels")
            
            # Load unified label mappings
            unified_mappings_path = self.model_dir / "unified_label_mappings.json"
            if unified_mappings_path.exists():
                with open(unified_mappings_path, 'r', encoding='utf-8') as f:
                    self.unified_mappings = json.load(f)
                logger.info(f"Loaded {len(self.unified_mappings)} unified label mappings")
            
            # Load current taxonomy
            taxonomy_path = Path("data/taxonomy/mood_food_taxonomy.json")
            if taxonomy_path.exists():
                with open(taxonomy_path, 'r', encoding='utf-8') as f:
                    taxonomy_data = json.load(f)
                self.taxonomy_categories = list(taxonomy_data.keys())
                logger.info(f"Loaded {len(self.taxonomy_categories)} taxonomy categories")
            
        except Exception as e:
            logger.error(f"Failed to load mappings: {e}")
    
    def _build_mapping_matrix(self):
        """Build intelligent mapping between original labels and new categories."""
        try:
            # Create semantic similarity mappings
            for original_label in self.original_labels:
                self.mapping_matrix[original_label] = self._find_semantic_matches(original_label)
            
            logger.info("Built taxonomy mapping matrix")
            
        except Exception as e:
            logger.error(f"Failed to build mapping matrix: {e}")
    
    def _find_semantic_matches(self, original_label: str) -> List[Tuple[str, float]]:
        """
        Find semantic matches for an original label in the new taxonomy.
        
        Returns list of (category_name, similarity_score) tuples.
        """
        matches = []
        
        # Parse the original label structure (e.g., "emotional_comfort" -> ["emotional", "comfort"])
        label_parts = original_label.split('_')
        
        for category in self.taxonomy_categories:
            score = self._calculate_semantic_similarity(original_label, category, label_parts)
            if score > 0.3:  # Threshold for meaningful similarity
                matches.append((category, score))
        
        # Sort by similarity score (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # Return top matches (limit to 5 to avoid overwhelming)
        return matches[:5]
    
    def _calculate_semantic_similarity(self, original_label: str, category: str, label_parts: List[str]) -> float:
        """
        Calculate semantic similarity between original label and new category.
        
        Uses multiple strategies:
        1. Exact substring matching
        2. Word overlap
        3. Semantic category matching
        """
        score = 0.0
        
        # Strategy 1: Exact substring matching
        if original_label.lower() in category.lower():
            score += 0.8
        elif category.lower() in original_label.lower():
            score += 0.6
        
        # Strategy 2: Word overlap
        category_words = set(category.lower().split('_'))
        label_words = set(label_parts)
        overlap = len(category_words.intersection(label_words))
        if overlap > 0:
            score += 0.4 * (overlap / max(len(category_words), len(label_words)))
        
        # Strategy 3: Semantic category matching
        semantic_mappings = {
            'emotional': ['mood', 'feeling', 'emotion', 'sentiment'],
            'energy': ['energetic', 'vitality', 'strength', 'power'],
            'flavor': ['taste', 'savor', 'flavor', 'spice'],
            'goal': ['purpose', 'aim', 'objective', 'target'],
            'health': ['wellness', 'nutrition', 'diet', 'fitness'],
            'occasion': ['event', 'celebration', 'meal', 'time'],
            'season': ['spring', 'summer', 'autumn', 'winter', 'seasonal'],
            'sensory': ['texture', 'temperature', 'sensation', 'feel'],
            'weather': ['climate', 'temperature', 'atmospheric', 'weather']
        }
        
        for semantic_key, related_words in semantic_mappings.items():
            if any(part in semantic_key for part in label_parts):
                if any(word in category.lower() for word in related_words):
                    score += 0.3
        
        return min(score, 1.0)  # Cap at 1.0
    
    def map_ml_prediction(self, ml_prediction: str, confidence: float) -> List[Tuple[str, float]]:
        """
        Map ML model prediction to relevant taxonomy categories.
        
        Args:
            ml_prediction: The predicted label from the ML model
            confidence: Confidence score from the ML model
            
        Returns:
            List of (category_name, adjusted_confidence) tuples
        """
        if ml_prediction not in self.mapping_matrix:
            logger.warning(f"Unknown ML prediction: {ml_prediction}")
            return []
        
        # Get the semantic matches
        matches = self.mapping_matrix[ml_prediction]
        
        # Adjust confidence scores based on semantic similarity
        adjusted_matches = []
        for category, similarity in matches:
            # Combine ML confidence with semantic similarity
            adjusted_confidence = confidence * similarity
            adjusted_matches.append((category, adjusted_confidence))
        
        return adjusted_matches
    
    def get_original_labels(self) -> List[str]:
        """Get the original 20 labels that the ML model was trained on."""
        return self.original_labels.copy()
    
    def get_taxonomy_categories(self) -> List[str]:
        """Get all 138 taxonomy categories."""
        return self.taxonomy_categories.copy()
    
    def get_mapping_stats(self) -> Dict[str, int | float]:
        """Get statistics about the mapping coverage."""
        total_mappings = sum(len(matches) for matches in self.mapping_matrix.values())
        avg_mappings = total_mappings / len(self.mapping_matrix) if self.mapping_matrix else 0
        
        return {
            "original_labels": len(self.original_labels),
            "taxonomy_categories": len(self.taxonomy_categories),
            "total_mappings": total_mappings,
            "average_mappings_per_label": round(avg_mappings, 2)
        }
    
    def print_mapping_sample(self, num_samples: int = 3):
        """Print a sample of the mappings for debugging."""
        print(f"\n📊 Taxonomy Mapping Sample (showing {num_samples} examples):")
        print("=" * 60)
        
        for i, (original_label, matches) in enumerate(self.mapping_matrix.items()):
            if i >= num_samples:
                break
            
            print(f"\n🔗 {original_label}:")
            for category, score in matches[:3]:  # Show top 3 matches
                print(f"   → {category} (similarity: {score:.2f})")
        
        print("\n" + "=" * 60)
