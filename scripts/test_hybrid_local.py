#!/usr/bin/env python3
"""
Local Test Script for Hybrid Intent Classification

This script tests the hybrid system without making actual LLM API calls.
It simulates LLM responses to verify the validation and comparison logic.
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from typing import List, Dict, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.nlu.validator import validate_labels
from utils.label_utils import load_taxonomy, load_dataset_labels

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LocalHybridTester:
    def __init__(self):
        """Initialize local tester with taxonomy and validation."""
        self.taxonomy_labels = self._load_taxonomy()
        self.auto_labeled_file = "data/auto_labeled_local.jsonl"
        
    def _load_taxonomy(self) -> List[str]:
        """Load taxonomy labels for validation."""
        try:
            import os
            from pathlib import Path
            
            # Try to load from unified mappings first
            mappings_path = Path("models/intent_classifier/unified_label_mappings.json")
            if mappings_path.exists():
                taxonomy = load_dataset_labels(mappings_path)
                if taxonomy:
                    logger.info(f"Loaded {len(taxonomy)} labels from unified mappings")
                    return taxonomy
            
            # Fallback to original taxonomy
            taxonomy_path = Path("data/taxonomy/mood_food_taxonomy.json")
            if taxonomy_path.exists():
                taxonomy = load_taxonomy(taxonomy_path)
                logger.info(f"Loaded {len(taxonomy)} labels from taxonomy")
                return taxonomy
            
            # Final fallback - hardcoded basic labels
            logger.warning("No taxonomy files found, using basic fallback labels")
            return [
                "goal_comfort", "sensory_warming", "meal_dinner", "emotional_comfort",
                "health_illness", "goal_hydration", "sensory_refreshing", "occasion_home"
            ]
            
        except Exception as e:
            logger.error(f"Failed to load taxonomy: {e}")
            # Return basic labels as fallback
            return [
                "goal_comfort", "sensory_warming", "meal_dinner", "emotional_comfort",
                "health_illness", "goal_hydration", "sensory_refreshing", "occasion_home"
            ]
    
    def simulate_llm_response(self, query: str) -> List[str]:
        """Simulate LLM response based on query content."""
        query_lower = query.lower()
        
        # Simple rule-based simulation
        labels = []
        
        if any(word in query_lower for word in ['warm', 'comfort', 'cozy', 'soothing']):
            labels.extend(['goal_comfort', 'sensory_warming'])
        
        if any(word in query_lower for word in ['dinner', 'lunch', 'breakfast']):
            if 'dinner' in query_lower:
                labels.append('meal_dinner')
            elif 'lunch' in query_lower:
                labels.append('meal_lunch')
            elif 'breakfast' in query_lower:
                labels.append('meal_breakfast')
        
        if any(word in query_lower for word in ['spicy', 'hot']):
            labels.append('flavor_spicy')
        
        if any(word in query_lower for word in ['sweet', 'dessert']):
            labels.append('flavor_sweet')
        
        if any(word in query_lower for word in ['ill', 'sick', 'recovery']):
            labels.extend(['health_illness', 'goal_comfort'])
        
        if any(word in query_lower for word in ['refreshing', 'cool', 'cold']):
            labels.extend(['sensory_refreshing', 'goal_hydration'])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_labels = []
        for label in labels:
            if label not in seen:
                seen.add(label)
                unique_labels.append(label)
        
        return unique_labels
    
    def simulate_ml_response(self, query: str) -> List[str]:
        """Simulate ML classifier response."""
        query_lower = query.lower()
        
        # Simulate ML classifier with different logic
        labels = []
        
        if any(word in query_lower for word in ['comfort', 'warm']):
            labels.append('comfort')
        
        if any(word in query_lower for word in ['health', 'ill', 'sick']):
            labels.append('health')
        
        if any(word in query_lower for word in ['energy', 'boost']):
            labels.append('energy')
        
        if any(word in query_lower for word in ['romantic', 'date']):
            labels.append('romantic')
        
        return labels
    
    def test_hybrid_local(self, query: str) -> Dict:
        """Test hybrid classification with simulated responses."""
        logger.info(f"=== Local Hybrid Test for: '{query}' ===")
        
        result = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "llm_labels": [],
            "validated_labels": [],
            "ml_labels": [],
            "comparison": {},
            "method": "local_simulation"
        }
        
        # Step 1: Simulate LLM Classification
        logger.info("Step 1: Simulated LLM Classification")
        llm_labels = self.simulate_llm_response(query)
        result["llm_labels"] = llm_labels
        logger.info(f"Simulated LLM labels: {llm_labels}")
        
        # Step 2: Validation
        logger.info("Step 2: Taxonomy Validation")
        validated_labels = validate_labels(llm_labels, self.taxonomy_labels)
        result["validated_labels"] = validated_labels
        logger.info(f"Validated labels: {validated_labels}")
        
        # Step 3: Simulate ML Classification
        logger.info("Step 3: Simulated ML Classification")
        ml_labels = self.simulate_ml_response(query)
        result["ml_labels"] = ml_labels
        logger.info(f"Simulated ML labels: {ml_labels}")
        
        # Step 4: Compare results
        result["comparison"] = self._compare_results(validated_labels, ml_labels)
        
        # Step 5: Log for dataset growth
        self._log_for_dataset_growth(result)
        
        return result
    
    def _compare_results(self, validated_labels: List[str], ml_labels: List[str]) -> Dict:
        """Compare LLM and ML classification results."""
        comparison = {
            "overlap": [],
            "llm_only": [],
            "ml_only": [],
            "agreement_score": 0.0
        }
        
        validated_set = set(validated_labels)
        ml_set = set(ml_labels)
        
        comparison["overlap"] = list(validated_set & ml_set)
        comparison["llm_only"] = list(validated_set - ml_set)
        comparison["ml_only"] = list(ml_set - validated_set)
        
        # Calculate agreement score
        total_labels = len(validated_set | ml_set)
        if total_labels > 0:
            overlap_count = len(comparison["overlap"])
            comparison["agreement_score"] = overlap_count / total_labels
        
        return comparison
    
    def _log_for_dataset_growth(self, result: Dict):
        """Log classification result for dataset growth."""
        try:
            # Create auto-labeled entry
            entry = {
                "text": result["query"],
                "labels": result["validated_labels"],
                "timestamp": result["timestamp"],
                "method": "local_simulation",
                "llm_labels": result["llm_labels"],
                "ml_labels": result.get("ml_labels", []),
                "comparison": result.get("comparison", {})
            }
            
            # Append to JSONL file
            with open(self.auto_labeled_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            logger.info(f"Logged result to {self.auto_labeled_file}")
            
        except Exception as e:
            logger.error(f"Failed to log result: {e}")
    
    def print_result(self, result: Dict):
        """Print classification result in a readable format."""
        print("\n" + "="*60)
        print(f"QUERY: {result['query']}")
        print("="*60)
        
        print(f"\n📊 Simulated LLM Classification:")
        print(f"   Raw LLM labels: {result['llm_labels']}")
        print(f"   Validated labels: {result['validated_labels']}")
        
        print(f"\n🤖 Simulated ML Classification:")
        print(f"   ML labels: {result['ml_labels']}")
        
        comparison = result.get("comparison", {})
        print(f"\n🔄 Comparison:")
        print(f"   Overlap: {comparison.get('overlap', [])}")
        print(f"   LLM only: {comparison.get('llm_only', [])}")
        print(f"   ML only: {comparison.get('ml_only', [])}")
        print(f"   Agreement score: {comparison.get('agreement_score', 0.0):.2f}")
        
        print(f"\n📝 Logged to: {self.auto_labeled_file}")
        print("="*60)

def main():
    """Main function for testing."""
    tester = LocalHybridTester()
    
    # Test queries
    test_queries = [
        "I want something warm and comforting for dinner",
        "I need something spicy and energizing for lunch",
        "I'm feeling ill and want something soothing",
        "I want a refreshing drink for hot weather",
        "I need something sweet for dessert"
    ]
    
    print("🧪 Local Hybrid Intent Classification Test")
    print("Testing validation and comparison logic without API calls")
    print("-" * 50)
    
    for query in test_queries:
        result = tester.test_hybrid_local(query)
        tester.print_result(result)
        print()

if __name__ == "__main__":
    main()
