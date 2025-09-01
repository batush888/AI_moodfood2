#!/usr/bin/env python3
"""
Script to integrate the newly trained model into the existing system.
"""

import os
import json
import shutil
from pathlib import Path
import subprocess
import sys

def backup_existing_model():
    """Backup the existing model before replacing it."""
    existing_model_dir = Path("models/intent_classifier")
    backup_dir = Path("models/intent_classifier_backup")
    
    if existing_model_dir.exists():
        print(f"📦 Backing up existing model to {backup_dir}")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(existing_model_dir, backup_dir)
        return True
    return False

def integrate_new_model():
    """Integrate the newly trained model into the system."""
    new_model_dir = Path("models/improved_intent_classifier")
    target_model_dir = Path("models/intent_classifier")
    
    if not new_model_dir.exists():
        print(f"❌ New model directory {new_model_dir} not found!")
        return False
    
    print(f"🔄 Integrating new model from {new_model_dir} to {target_model_dir}")
    
    # Remove existing model directory
    if target_model_dir.exists():
        shutil.rmtree(target_model_dir)
    
    # Copy new model
    shutil.copytree(new_model_dir, target_model_dir)
    
    print(f"✅ Model integrated successfully!")
    return True

def update_model_config():
    """Update model configuration files if needed."""
    config_files = [
        "core/nlu/enhanced_intent_classifier.py",
        "api/enhanced_routes.py"
    ]
    
    print("🔧 Checking model configuration files...")
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"   ✓ {config_file} exists")
        else:
            print(f"   ⚠️ {config_file} not found")

def test_model_integration():
    """Test that the integrated model works correctly."""
    print("🧪 Testing model integration...")
    
    try:
        # Test import
        sys.path.append('.')
        from core.nlu.enhanced_intent_classifier import EnhancedIntentClassifier
        
        # Initialize classifier
        classifier = EnhancedIntentClassifier(
            taxonomy_path="data/taxonomy/mood_food_taxonomy.json",
            model_dir="models/intent_classifier"
        )
        
        # Test prediction
        test_query = "I want something spicy and bold"
        result = classifier.classify_intent(test_query)
        
        print(f"   ✓ Model loaded successfully")
        print(f"   ✓ Test query: '{test_query}'")
        print(f"   ✓ Prediction: {result}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model integration test failed: {e}")
        return False

def create_model_summary():
    """Create a summary of the model training and integration."""
    model_dir = Path("models/intent_classifier")
    
    if not model_dir.exists():
        print("❌ Model directory not found!")
        return
    
    # Load training info
    training_info_path = model_dir / "training_info.json"
    if training_info_path.exists():
        with open(training_info_path, 'r') as f:
            training_info = json.load(f)
        
        summary = {
            "model_integration_summary": {
                "timestamp": str(Path().cwd()),
                "model_directory": str(model_dir),
                "training_info": training_info,
                "integration_status": "completed"
            }
        }
        
        # Save summary
        summary_path = Path("models/model_integration_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Model integration summary saved to {summary_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("🎯 MODEL INTEGRATION SUMMARY")
        print("="*80)
        print(f"📁 Model Directory: {model_dir}")
        print(f"🎯 Number of Labels: {training_info.get('num_labels', 'Unknown')}")
        print(f"📊 Training Samples: {training_info.get('total_samples', 'Unknown')}")
        print(f"🏋️ Training Epochs: {training_info.get('epochs', 'Unknown')}")
        print(f"📈 Best Validation Loss: {training_info.get('best_val_loss', 'Unknown')}")
        print(f"🔧 Batch Size: {training_info.get('batch_size', 'Unknown')}")
        print(f"📚 Learning Rate: {training_info.get('learning_rate', 'Unknown')}")
        print("="*80)

def main():
    """Main integration process."""
    print("🚀 Starting model integration process...")
    
    # Step 1: Backup existing model
    backup_existing_model()
    
    # Step 2: Integrate new model
    if not integrate_new_model():
        print("❌ Model integration failed!")
        return False
    
    # Step 3: Update configuration
    update_model_config()
    
    # Step 4: Test integration
    if not test_model_integration():
        print("❌ Model integration test failed!")
        return False
    
    # Step 5: Create summary
    create_model_summary()
    
    print("\n🎉 Model integration completed successfully!")
    print("🔄 You may need to restart the server for changes to take effect.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
