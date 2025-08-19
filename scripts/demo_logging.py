#!/usr/bin/env python3
"""
Demonstration Script for Comprehensive Query Logging System
Shows how the system automatically logs all queries and grows the dataset
"""

import json
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_logging_system():
    """Demonstrate the comprehensive logging system"""
    
    print("🚀 MoodFood AI - Comprehensive Query Logging System Demo")
    print("=" * 60)
    
    # Check if logging system is available
    try:
        from core.logging.query_logger import query_logger
        print("✅ Query logging system imported successfully")
    except ImportError as e:
        print(f"❌ Query logging system not available: {e}")
        print("Please ensure the logging module is properly installed")
        return
    
    # Demonstrate logging a query
    print("\n📝 Demonstrating Query Logging...")
    
    # Simulate a user query
    sample_query = "I want something warm and comforting for dinner"
    user_context = {
        "time_of_day": "evening",
        "weather": "cold",
        "mood": "tired",
        "social_context": "alone"
    }
    
    # Log the query
    query_id = query_logger.log_query(
        text_input=sample_query,
        user_context=user_context,
        session_id="demo_session_001"
    )
    
    print(f"📋 Query logged with ID: {query_id}")
    
    # Simulate intent classification results
    print("\n🧠 Logging Intent Classification Results...")
    
    intent_results = {
        "primary_intent": "goal_comfort",
        "confidence": 0.85,
        "all_intents": [["goal_comfort", 0.85], ["sensory_warming", 0.78]],
        "method": "hybrid_llm"
    }
    
    query_logger.log_intent_results(
        query_id=query_id,
        primary_intent=intent_results["primary_intent"],
        confidence=intent_results["confidence"],
        all_intents=intent_results["all_intents"],
        method=intent_results["method"],
        processing_time_ms=150.0
    )
    
    print(f"✅ Intent results logged for query: {query_id}")
    
    # Simulate LLM results
    print("\n🤖 Logging LLM Classification Results...")
    
    llm_labels = ["goal_comfort", "sensory_warming", "meal_dinner"]
    validated_labels = ["goal_comfort", "sensory_warming"]
    ml_labels = ["comfort"]
    
    query_logger.log_llm_results(
        query_id=query_id,
        llm_labels=llm_labels,
        validated_labels=validated_labels,
        ml_labels=ml_labels,
        comparison_score=0.75
    )
    
    print(f"✅ LLM results logged for query: {query_id}")
    
    # Simulate recommendation results
    print("\n🍽️ Logging Recommendation Results...")
    
    recommendations = [
        {
            "food_name": "chicken soup",
            "food_category": "SOUP",
            "score": 0.9,
            "mood_match": 0.85
        },
        {
            "food_name": "hot chocolate",
            "food_category": "BEVERAGE",
            "score": 0.8,
            "mood_match": 0.78
        }
    ]
    
    query_logger.log_recommendations(
        query_id=query_id,
        recommendations=recommendations,
        engine_time_ms=200.0
    )
    
    print(f"✅ Recommendations logged for query: {query_id}")
    
    # Show logging statistics
    print("\n📊 Current Logging Statistics...")
    
    stats = query_logger.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    # Demonstrate export for training
    print("\n📤 Exporting High-Quality Data for Training...")
    
    training_file = "data/demo_training_dataset.jsonl"
    exported_count = query_logger.export_for_training(training_file)
    
    print(f"✅ Exported {exported_count} high-quality entries to {training_file}")
    
    # Show the exported training data
    if Path(training_file).exists():
        print(f"\n📖 Sample of exported training data:")
        with open(training_file, 'r') as f:
            for i, line in enumerate(f):
                if i < 3:  # Show first 3 entries
                    data = json.loads(line.strip())
                    print(f"   Entry {i+1}: {data['text'][:50]}... → {data['labels']}")
                else:
                    break
    
    print("\n🎯 Logging System Features Demonstrated:")
    print("   ✅ Automatic query logging")
    print("   ✅ Intent classification tracking")
    print("   ✅ LLM vs ML comparison logging")
    print("   ✅ Recommendation result logging")
    print("   ✅ Performance metrics tracking")
    print("   ✅ Error logging and debugging")
    print("   ✅ Training dataset export")
    print("   ✅ Statistics and monitoring")
    
    print("\n💡 How This Grows Your Dataset:")
    print("   1. Every user query is automatically logged")
    print("   2. All classification results are captured")
    print("   3. Success/failure patterns are tracked")
    print("   4. High-quality entries are exported for training")
    print("   5. Continuous improvement through data analysis")

def show_log_file_structure():
    """Show the structure of the logging files"""
    
    print("\n📁 Logging File Structure:")
    print("=" * 40)
    
    log_dir = Path("data/logs")
    if log_dir.exists():
        print(f"📂 Log directory: {log_dir}")
        
        for file_path in log_dir.glob("*.jsonl"):
            size = file_path.stat().st_size
            print(f"   📄 {file_path.name}: {size} bytes")
            
            # Show sample content
            try:
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        print(f"      Sample: {list(data.keys())}")
            except Exception as e:
                print(f"      Error reading: {e}")
    else:
        print("❌ Log directory not found")
    
    # Show auto_labeled.jsonl
    auto_labeled = Path("data/auto_labeled.jsonl")
    if auto_labeled.exists():
        size = auto_labeled.stat().st_size
        print(f"📄 {auto_labeled.name}: {size} bytes")
    else:
        print("❌ auto_labeled.jsonl not found")

def demonstrate_api_endpoints():
    """Show the available logging API endpoints"""
    
    print("\n🌐 Available Logging API Endpoints:")
    print("=" * 40)
    
    endpoints = [
        ("GET", "/logging/stats", "Get comprehensive logging statistics"),
        ("POST", "/logging/export-training", "Export high-quality data for training"),
        ("GET", "/logging/query/{query_id}", "Get detailed log for specific query")
    ]
    
    for method, endpoint, description in endpoints:
        print(f"   {method:6} {endpoint:<30} {description}")

def main():
    """Main demonstration function"""
    
    try:
        demonstrate_logging_system()
        show_log_file_structure()
        demonstrate_api_endpoints()
        
        print("\n🎉 Logging System Demo Complete!")
        print("\n📚 Next Steps:")
        print("   1. Start your API server: python api/enhanced_routes.py")
        print("   2. Use the web interface to make queries")
        print("   3. Check logs at: data/logs/query_logs.jsonl")
        print("   4. Export training data: curl -X POST http://localhost:8000/logging/export-training")
        print("   5. View statistics: curl http://localhost:8000/logging/stats")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
