#!/usr/bin/env python3
"""
Minimal test server to test the hybrid filter functionality.
"""

import asyncio
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import the hybrid filter components
from core.filtering.hybrid_filter import HybridFilter
from core.filtering.llm_validator import LLMValidator
from core.nlu.enhanced_intent_classifier import EnhancedIntentClassifier

app = FastAPI(title="Hybrid Filter Test Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
hybrid_filter = None
enhanced_classifier = None

@app.on_event("startup")
async def startup_event():
    """Initialize the hybrid filter on startup."""
    global hybrid_filter, enhanced_classifier
    
    print("🚀 Starting Hybrid Filter Test Server...")
    
    try:
        # Initialize enhanced classifier
        print("Initializing Enhanced Intent Classifier...")
        taxonomy_path = "data/taxonomy/mood_food_taxonomy.json"
        enhanced_classifier = EnhancedIntentClassifier(taxonomy_path)
        print("✅ Enhanced Intent Classifier initialized")
        
        # Initialize LLM validator
        print("Initializing LLM Validator...")
        llm_validator = LLMValidator()
        print(f"✅ LLM Validator initialized (enabled: {llm_validator.enabled})")
        
        # Initialize hybrid filter
        print("Initializing Hybrid Filter...")
        hybrid_filter = HybridFilter(
            llm_validator=llm_validator,
            ml_classifier=enhanced_classifier,
            confidence_threshold=0.7
        )
        print("✅ Hybrid Filter initialized")
        
        print("🎉 All components initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        import traceback
        traceback.print_exc()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if hybrid_filter else "degraded",
        "components": {
            "enhanced_classifier": "operational" if enhanced_classifier else "not_available",
            "hybrid_filter": "operational" if hybrid_filter else "not_available"
        },
        "hybrid_filter_stats": hybrid_filter.get_live_stats() if hybrid_filter else None
    }

@app.post("/test-hybrid")
async def test_hybrid_filter(request: dict):
    """Test the hybrid filter with a query."""
    if not hybrid_filter:
        raise HTTPException(status_code=503, detail="Hybrid filter not available")
    
    try:
        query = request.get("query", "I want some Japanese food")
        user_context = request.get("user_context", {})
        
        print(f"🧪 Testing hybrid filter with query: '{query}'")
        
        # Process query through hybrid filter
        result = await hybrid_filter.process_query(query, user_context)
        
        # Convert result to dict for JSON serialization
        response = {
            "decision": result.decision,
            "recommendations": result.recommendations,
            "reasoning": result.reasoning,
            "processing_time_ms": result.processing_time_ms,
            "timestamp": result.timestamp,
            "ml_prediction": {
                "primary_intent": result.ml_prediction.primary_intent,
                "confidence": result.ml_prediction.confidence,
                "method": result.ml_prediction.method
            } if result.ml_prediction else None,
            "llm_interpretation": {
                "intent": result.llm_interpretation.intent,
                "reasoning": result.llm_interpretation.reasoning,
                "confidence": result.llm_interpretation.confidence
            } if result.llm_interpretation else None
        }
        
        print(f"✅ Query processed successfully. Decision: {result.decision}")
        print(f"   Recommendations: {len(result.recommendations)} items")
        print(f"   Processing time: {result.processing_time_ms:.2f}ms")
        
        return response
        
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/hybrid-stats")
async def get_hybrid_stats():
    """Get hybrid filter statistics."""
    if not hybrid_filter:
        raise HTTPException(status_code=503, detail="Hybrid filter not available")
    
    return hybrid_filter.get_live_stats()

if __name__ == "__main__":
    print("🧪 Starting Hybrid Filter Test Server...")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
