#!/usr/bin/env python3
"""
Demonstration: How to Integrate Live Stats into Your Inference Pipeline
"""

from core.filtering.global_filter import update_global_filter_stats
import time
import random

def simulate_inference_pipeline():
    """
    This is how you would integrate live stats into your actual inference pipeline.
    Replace the simulation logic with your real inference code.
    """
    
    print("🚀 Simulating Inference Pipeline with Live Stats Integration")
    print("=" * 65)
    
    # Simulate processing multiple user queries
    queries = [
        "I want comfort food for dinner",
        "Something light and refreshing",
        "Warm and spicy meal",
        "Healthy breakfast options",
        "Quick lunch ideas",
        "Romantic dinner suggestions",
        "Food for a party",
        "Energy-boosting snacks"
    ]
    
    print(f"📝 Processing {len(queries)} user queries...")
    
    for i, query in enumerate(queries, 1):
        print(f"\n🔍 Processing Query {i}: '{query}'")
        
        # Simulate your inference logic here
        # This is where you'd call your actual ML classifier, LLM parser, etc.
        
        # Simulate confidence scores (replace with real ML confidence)
        ml_confidence = random.uniform(0.3, 0.9)
        
        # Simulate decision making based on confidence
        if ml_confidence >= 0.7:
            decision = "ml_confident"
            print(f"  ✅ ML confident (confidence: {ml_confidence:.2f})")
        elif ml_confidence >= 0.45:
            decision = "llm_fallback"
            print(f"  ⚠️  LLM fallback (confidence: {ml_confidence:.2f})")
        else:
            decision = "rejected"
            print(f"  ❌ Rejected (confidence: {ml_confidence:.2f})")
        
        # 🔑 KEY INTEGRATION POINT: Update live stats
        update_global_filter_stats(decision)
        
        # Simulate processing time
        time.sleep(0.5)
    
    print(f"\n🎯 Inference Pipeline Complete!")
    print(f"   All queries processed with live stats tracking.")

def show_integration_examples():
    """Show code examples for integrating live stats"""
    
    print("\n📚 Integration Examples")
    print("=" * 30)
    
    print("""
🔑 **Key Integration Points:**

1. **In your ML classifier:**
   ```python
   from core.filtering.global_filter import update_global_filter_stats
   
   def classify_intent(query):
       confidence = ml_model.predict_proba([query]).max()
       
       if confidence >= 0.7:
           update_global_filter_stats("ml_confident")
           return ml_predictions
       elif confidence >= 0.45:
           update_global_filter_stats("llm_fallback")
           return llm_fallback_predictions
       else:
           update_global_filter_stats("rejected")
           return fallback_predictions
   ```

2. **In your LLM parser:**
   ```python
   def parse_with_llm(query):
       try:
           result = llm_api.parse(query)
           update_global_filter_stats("llm_fallback")
           return result
       except Exception:
           update_global_filter_stats("rejected")
           return None
   ```

3. **In your main inference function:**
   ```python
   def get_recommendations(query):
       # Your existing logic here
       
       # Track the final decision
       if final_confidence >= 0.7:
           update_global_filter_stats("ml_confident")
       elif used_llm_fallback:
           update_global_filter_stats("llm_fallback")
       else:
           update_global_filter_stats("rejected")
       
       return recommendations
   ```

4. **Monitor live stats:**
   ```python
   from core.filtering.global_filter import get_global_filter_live_stats
   
   # Get current stats
   current_stats = get_global_filter_live_stats()
   print("ML: {}, LLM: {}, Rejected: {}".format(
       current_stats['ml_confident'], 
       current_stats['llm_fallback'], 
       current_stats['rejected']
   ))
   ```
""")

if __name__ == "__main__":
    # Run the simulation
    simulate_inference_pipeline()
    
    # Show integration examples
    show_integration_examples()
    
    print(f"\n🎉 **Next Steps:**")
    print(f"   1. Replace simulation logic with your real inference code")
    print(f"   2. Add update_global_filter_stats() calls at decision points")
    print(f"   3. View live stats in monitoring.html dashboard")
    print(f"   4. Monitor performance in real-time!")
