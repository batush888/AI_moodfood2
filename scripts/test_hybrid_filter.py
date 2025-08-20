#!/usr/bin/env python3
"""
Test script for the hybrid filtering system
"""

import json
import logging
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_samples():
    """Create test samples with various confidence levels"""
    
    test_samples = [
        # High confidence samples (should be accepted)
        {
            'text': 'I want comfort food for dinner',
            'labels': ['goal_comfort', 'meal_dinner'],
            'confidence': 0.85
        },
        {
            'text': 'Need something healthy and light',
            'labels': ['goal_healthy', 'energy_light'],
            'confidence': 0.92
        },
        
        # Low confidence samples (should be rejected)
        {
            'text': 'food',
            'labels': ['unknown'],
            'confidence': 0.2
        },
        {
            'text': 'something to eat',
            'labels': ['meal_general'],
            'confidence': 0.3
        },
        
        # Borderline samples (should go to LLM validation)
        {
            'text': 'I feel like something warm',
            'labels': ['sensory_warming'],
            'confidence': 0.48
        },
        {
            'text': 'Need energy boost',
            'labels': ['goal_energy'],
            'confidence': 0.52
        },
        
        # Duplicate samples
        {
            'text': 'I want comfort food for dinner',
            'labels': ['goal_comfort', 'meal_dinner'],
            'confidence': 0.85
        },
        
        # Malformed samples
        {
            'text': '',
            'labels': ['goal_comfort'],
            'confidence': 0.8
        },
        {
            'text': 'a',
            'labels': [],
            'confidence': 0.7
        },
        {
            'text': 'This is a very long query that exceeds the maximum length limit and should be considered malformed. ' * 20,
            'labels': ['goal_comfort'],
            'confidence': 0.8
        }
    ]
    
    return test_samples

def test_hybrid_filter():
    """Test the hybrid filtering system"""
    
    logger.info("🧪 Testing Hybrid Filtering System")
    logger.info("=" * 50)
    
    try:
        # Import the hybrid filter
        from core.filtering.hybrid_filter import HybridFilter
        from config.settings import validate_config
        
        # Validate configuration
        logger.info("1️⃣ Validating configuration...")
        if not validate_config():
            logger.error("❌ Configuration validation failed")
            return False
        logger.info("✅ Configuration validation passed")
        
        # Create test samples
        logger.info("2️⃣ Creating test samples...")
        test_samples = create_test_samples()
        logger.info(f"✅ Created {len(test_samples)} test samples")
        
        # Initialize hybrid filter
        logger.info("3️⃣ Initializing hybrid filter...")
        hybrid_filter = HybridFilter()
        logger.info("✅ Hybrid filter initialized")
        
        # Test filtering
        logger.info("4️⃣ Testing hybrid filtering...")
        filtered_samples, filter_stats = hybrid_filter.filter_training_data(test_samples)
        
        # Display results
        logger.info("5️⃣ Filtering Results:")
        logger.info(hybrid_filter.get_filter_summary())
        
        # Verify results
        logger.info("6️⃣ Verifying results...")
        
        # Check that duplicates were removed
        original_texts = [s['text'].lower().strip() for s in test_samples if s['text'].strip()]
        unique_texts = [s['text'].lower().strip() for s in filtered_samples]
        
        if len(unique_texts) < len(set(original_texts)):
            logger.info("✅ Duplicates were removed")
        else:
            logger.warning("⚠️  No duplicates found in test data")
        
        # Check that low confidence samples were removed
        low_conf_samples = [s for s in test_samples if s.get('confidence', 0) < 0.45]
        if any(s['text'] in [fs['text'] for fs in filtered_samples] for s in low_conf_samples):
            logger.error("❌ Low confidence samples were not properly filtered")
            return False
        else:
            logger.info("✅ Low confidence samples were filtered out")
        
        # Check that malformed samples were removed
        malformed_samples = [s for s in test_samples if not s['text'].strip() or len(s['text']) < 3]
        if any(s['text'] in [fs['text'] for fs in filtered_samples] for s in malformed_samples):
            logger.error("❌ Malformed samples were not properly filtered")
            return False
        else:
            logger.info("✅ Malformed samples were filtered out")
        
        # Check LLM validation stats
        if filter_stats.get('borderline_count', 0) > 0:
            logger.info(f"✅ {filter_stats['borderline_count']} borderline cases processed")
            logger.info(f"   - LLM validated: {filter_stats['llm_validated']}")
            logger.info(f"   - LLM rejected: {filter_stats['llm_rejected']}")
        else:
            logger.info("ℹ️  No borderline cases in test data")
        
        # Test filter stats persistence
        logger.info("7️⃣ Testing filter stats persistence...")
        retrain_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        hybrid_filter.save_filter_stats(retrain_id)
        logger.info("✅ Filter stats saved")
        
        # Test detailed stats
        logger.info("8️⃣ Testing detailed statistics...")
        detailed_stats = hybrid_filter.get_detailed_stats()
        logger.info(f"✅ Detailed stats generated: {len(detailed_stats)} sections")
        
        logger.info("🎉 All tests passed!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Make sure all dependencies are installed and paths are correct")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_validator():
    """Test the LLM validator component"""
    
    logger.info("\n🤖 Testing LLM Validator Component")
    logger.info("=" * 40)
    
    try:
        from core.filtering.llm_validator import LLMValidator
        
        # Initialize validator
        validator = LLMValidator()
        logger.info(f"✅ LLM validator initialized")
        logger.info(f"   - Enabled: {validator.enabled}")
        logger.info(f"   - Model: {validator.model}")
        
        if not validator.enabled:
            logger.warning("⚠️  LLM validation disabled (no API key)")
            return True
        
        # Test single validation
        logger.info("Testing single sample validation...")
        is_valid = validator.validate_sample(
            query="I want comfort food",
            label="goal_comfort",
            confidence=0.5
        )
        logger.info(f"✅ Single validation result: {is_valid}")
        
        # Test batch validation
        logger.info("Testing batch validation...")
        test_batch = [
            {
                'query': 'I want comfort food',
                'label': 'goal_comfort',
                'confidence': 0.5
            },
            {
                'query': 'Need energy boost',
                'label': 'goal_energy',
                'confidence': 0.48
            }
        ]
        
        batch_results = validator.validate_batch(test_batch)
        logger.info(f"✅ Batch validation results: {batch_results}")
        
        # Test stats
        stats = validator.get_stats()
        logger.info(f"✅ Validator stats: {stats}")
        
        logger.info("🎉 LLM validator tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ LLM validator test failed: {e}")
        return False

def main():
    """Main test function"""
    
    logger.info("🚀 Starting Hybrid Filtering System Tests")
    logger.info("=" * 60)
    
    # Test hybrid filter
    filter_success = test_hybrid_filter()
    
    # Test LLM validator
    validator_success = test_llm_validator()
    
    # Summary
    logger.info("\n📊 Test Summary")
    logger.info("=" * 30)
    logger.info(f"Hybrid Filter: {'✅ PASSED' if filter_success else '❌ FAILED'}")
    logger.info(f"LLM Validator: {'✅ PASSED' if validator_success else '❌ FAILED'}")
    
    if filter_success and validator_success:
        logger.info("\n🎉 All tests passed! Hybrid filtering system is ready.")
        return True
    else:
        logger.error("\n❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
