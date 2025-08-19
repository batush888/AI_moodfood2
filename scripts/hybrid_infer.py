#!/usr/bin/env python3
"""
Hybrid Intent Classification Script

This script demonstrates the hybrid approach:
1. LLM semantic parsing (DeepSeek)
2. Taxonomy validation
3. Optional comparison with ML classifier
4. Data logging for dataset growth
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

from core.nlu.llm_parser import classify_with_llm
from core.nlu.validator import validate_labels
from utils.label_utils import load_taxonomy, load_dataset_labels
from scripts.infer_intent import predict_intents

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybridIntentClassifier:
    def __init__(self):
        """Initialize hybrid classifier with taxonomy and validation."""
        self.taxonomy_labels = self._load_taxonomy()
        self.auto_labeled_file = "data/auto_labeled.jsonl"
        
    def _load_taxonomy(self) -> List[str]:
        """Load taxonomy labels for validation."""
        try:
            # Try to load from unified mappings first
            import os
            mappings_path = os.path.join("models", "intent_classifier", "unified_label_mappings.json")
            if os.path.exists(mappings_path):
                taxonomy = load_dataset_labels(mappings_path)
                if taxonomy:
                    logger.info(f"Loaded {len(taxonomy)} labels from unified mappings")
                    return taxonomy
            
            # Fallback to original taxonomy
            taxonomy_path = os.path.join("data", "taxonomy", "mood_food_taxonomy.json")
            if os.path.exists(taxonomy_path):
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
    
    async def classify_hybrid(self, query: str, compare_ml: bool = True) -> Dict:
        """Perform hybrid intent classification."""
        logger.info(f"=== Hybrid Classification for: '{query}' ===")
        
        result = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "llm_labels": [],
            "validated_labels": [],
            "ml_labels": [],
            "comparison": {},
            "method": "hybrid"
        }
        
        # Step 1: LLM Classification
        logger.info("Step 1: LLM Semantic Parsing")
        try:
            llm_labels = await classify_with_llm(query, self.taxonomy_labels)
            result["llm_labels"] = llm_labels
            logger.info(f"LLM labels: {llm_labels}")
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            result["llm_labels"] = []
        
        # Step 2: Validation
        logger.info("Step 2: Taxonomy Validation")
        try:
            validated_labels = validate_labels(llm_labels, self.taxonomy_labels)
            result["validated_labels"] = validated_labels
            logger.info(f"Validated labels: {validated_labels}")
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            result["validated_labels"] = []
        
        # Step 3: Optional ML Comparison
        if compare_ml:
            logger.info("Step 3: ML Classifier Comparison")
            try:
                ml_result = predict_intents(query)
                if ml_result and isinstance(ml_result, dict) and "primary_intent" in ml_result:
                    ml_labels = [ml_result["primary_intent"]]
                    if "all_intents" in ml_result:
                        # Extract labels from all_intents
                        all_intents = ml_result["all_intents"]
                        if isinstance(all_intents, list):
                            for intent in all_intents:
                                if isinstance(intent, (list, tuple)) and len(intent) > 0:
                                    ml_labels.append(intent[0])
                                elif isinstance(intent, str):
                                    ml_labels.append(intent)
                    
                    result["ml_labels"] = ml_labels
                    logger.info(f"ML labels: {ml_labels}")
                    
                    # Compare results
                    result["comparison"] = self._compare_results(
                        validated_labels, ml_labels
                    )
                else:
                    logger.warning("ML classification returned no results")
                    result["ml_labels"] = []
                    result["comparison"] = self._compare_results(validated_labels, [])
                    
            except Exception as e:
                logger.error(f"ML classification failed: {e}")
                result["ml_labels"] = []
                result["comparison"] = self._compare_results(validated_labels, [])
        
        # Step 4: Log for dataset growth
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
                "method": "hybrid_llm",
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
        
        print(f"\n📊 LLM Classification:")
        print(f"   Raw LLM labels: {result['llm_labels']}")
        print(f"   Validated labels: {result['validated_labels']}")
        
        if result.get("ml_labels"):
            print(f"\n🤖 ML Classification:")
            print(f"   ML labels: {result['ml_labels']}")
            
            comparison = result.get("comparison", {})
            print(f"\n🔄 Comparison:")
            print(f"   Overlap: {comparison.get('overlap', [])}")
            print(f"   LLM only: {comparison.get('llm_only', [])}")
            print(f"   ML only: {comparison.get('ml_only', [])}")
            print(f"   Agreement score: {comparison.get('agreement_score', 0.0):.2f}")
        
        print(f"\n📝 Logged to: {self.auto_labeled_file}")
        print("="*60)

async def main():
    """Main function for interactive testing."""
    classifier = HybridIntentClassifier()
    
    if len(sys.argv) > 1:
        # Command line mode
        query = " ".join(sys.argv[1:])
        result = await classifier.classify_hybrid(query)
        classifier.print_result(result)
    else:
        # Interactive mode
        print("🤖 Hybrid Intent Classifier (LLM + Validation + ML)")
        print("Enter queries to classify (or 'quit' to exit)")
        print("-" * 50)
        
        while True:
            try:
                query = input("\nEnter query: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                result = await classifier.classify_hybrid(query)
                classifier.print_result(result)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
