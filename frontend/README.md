# Frontend-Backend API Contract

This document maps each frontend HTML file to its corresponding backend API endpoints and data contracts.

## 📁 **File Structure**

```
frontend/
├── index.html                    # Main unified frontend (recommended)
├── enhanced_ui.html             # Advanced features frontend
├── enhanced_ui_simple.html      # Simplified version
├── debug_test.html              # Testing/debugging interface
└── README.md                    # This file
```

## 🔗 **API Endpoint Mapping**

### **Main Frontend (index.html)**
- **Primary Endpoint**: `/enhanced-recommend` (POST)
- **Health Check**: `/health` (GET)
- **Examples**: `/examples` (GET)
- **Feedback**: `/feedback` (POST)

### **Enhanced UI (enhanced_ui.html)**
- **Primary Endpoint**: `/enhanced-recommend` (POST)
- **Health Check**: `/health` (GET)
- **Examples**: `/examples` (GET)
- **Feedback**: `/feedback` (POST)

### **Simple UI (enhanced_ui_simple.html)**
- **Primary Endpoint**: `/enhanced-recommend` (POST)
- **Health Check**: `/health` (GET)

### **Debug Test (debug_test.html)**
- **Test Endpoint**: `/test` (GET)
- **Test Recommendations**: `/test-recommendations` (POST)

## 📊 **Data Contracts**

### **Enhanced Recommendation Request**
```json
{
  "text_input": "string",
  "image_base64": "string|null",
  "audio_base64": "string|null",
  "user_context": {
    "time_of_day": "string",
    "weather": "string",
    "social_context": "string",
    "energy_level": "string"
  }
}
```

### **Enhanced Recommendation Response**
```json
{
  "recommendations": [
    {
      "food_name": "string",
      "food_category": "string",
      "food_region": "string",
      "food_culture": "string",
      "score": "float",
      "mood_match": "float",
      "reasoning": ["string"],
      "restaurant": {
        "name": "string",
        "rating": "float",
        "cuisine_type": "string",
        "delivery_available": "boolean"
      }
    }
  ],
  "intent_prediction": {
    "primary_intent": "string",
    "confidence": "float",
    "all_intents": [["string", "float"]]
  },
  "multimodal_analysis": {
    "combined_confidence": "float",
    "mood_categories": ["string"]
  },
  "processing_time": "float",
  "model_version": "string",
  "system_performance": {
    "phase_times_ms": {
      "intent_ms": "float",
      "multimodal_ms": "float",
      "engine_ms": "float",
      "format_ms": "float"
    },
    "timeouts": {
      "intent": "boolean",
      "multimodal": "boolean",
      "engine": "boolean"
    },
    "input_sizes": {
      "text_len": "integer",
      "image_b64_len": "integer",
      "audio_b64_len": "integer"
    }
  }
}
```

### **Feedback Request**
```json
{
  "user_id": "string",
  "session_id": "string",
  "input_text": "string",
  "recommended_foods": ["string"],
  "selected_food": "string|null",
  "rating": "float|null",
  "feedback_text": "string|null",
  "context": "object|null",
  "model_version": "string"
}
```

## 🚨 **Troubleshooting Guide**

### **Frontend Hangs at "Loading..."**
1. Check browser console for JavaScript errors
2. Verify API endpoint is accessible (`/health`)
3. Check backend logs for request processing
4. Verify request payload matches expected schema

### **API Returns 500 Error**
1. Check backend logs for Python exceptions
2. Verify all required fields are present in request
3. Check if AI components are properly initialized
4. Look for Pydantic validation errors

### **No Recommendations Generated**
1. Check if intent classification succeeded
2. Verify recommendation engine is operational
3. Check taxonomy file exists and is valid
4. Look for fallback recommendation logic

## 🔧 **Development Notes**

- **Base URL**: All frontends use `window.location.origin` for dynamic API resolution
- **CORS**: Backend serves frontend from same origin to avoid CORS issues
- **Error Handling**: Frontend includes comprehensive error handling and user feedback
- **Performance**: API includes detailed timing metrics for debugging
- **Fallbacks**: System gracefully degrades when AI components fail

## 📈 **Future Enhancements**

- **Real-time Updates**: WebSocket integration for live recommendations
- **User Profiles**: Persistent user preferences and history
- **A/B Testing**: Multiple recommendation algorithms
- **Analytics Dashboard**: User behavior and system performance metrics
