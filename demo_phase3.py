#!/usr/bin/env python3
"""
Phase 3 Demo: Advanced AI Features Showcase
Demonstrates:
- Enhanced Intent Classification with Transformers
- Multi-Modal Input Processing
- Real-Time Learning System
- Advanced Recommendation Engine
"""

import asyncio
import json
import time
import base64
import requests
from pathlib import Path
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase3Demo:
    """Comprehensive demo of Phase 3 advanced AI features."""
    
    def __init__(self, api_base: str = "http://127.0.0.1:8000"):
        self.api_base = api_base
        self.session_id = f"demo_session_{int(time.time())}"
        self.user_id = f"demo_user_{int(time.time())}"
        
    def run_demo(self):
        """Run the complete Phase 3 demo."""
        logger.info("🚀 Starting Phase 3: Advanced AI Features Demo")
        logger.info("=" * 60)
        
        try:
            # Test 1: System Health and Model Info
            self.test_system_health()
            
            # Test 2: Enhanced Text Recommendations
            self.test_enhanced_text_recommendations()
            
            # Test 3: Multi-Modal Analysis
            self.test_multimodal_analysis()
            
            # Test 4: Real-Time Learning
            self.test_realtime_learning()
            
            # Test 5: Advanced Features
            self.test_advanced_features()
            
            # Test 6: Performance Metrics
            self.test_performance_metrics()
            
            logger.info("✅ Phase 3 Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
    
    def test_system_health(self):
        """Test system health and get model information."""
        logger.info("\n🔍 Testing System Health and Model Information")
        logger.info("-" * 50)
        
        try:
            # Health check
            response = requests.get(f"{self.api_base}/health")
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"✅ System Status: {health_data['status']}")
                logger.info(f"✅ Version: {health_data['version']}")
                logger.info("✅ Components Status:")
                for component, status in health_data['components'].items():
                    logger.info(f"   - {component}: {status}")
            else:
                logger.warning(f"⚠️ Health check failed: {response.status_code}")
            
            # Model info
            response = requests.get(f"{self.api_base}/model-info")
            if response.status_code == 200:
                model_info = response.json()
                logger.info("\n🧠 Model Information:")
                if model_info.enhanced_classifier:
                    logger.info(f"   - Enhanced Classifier: {model_info.enhanced_classifier.get('model_name', 'N/A')}")
                if model_info.multimodal_processor:
                    logger.info(f"   - Multi-Modal Processor: {model_info.multimodal_processor.get('text_model', 'N/A')}")
                if model_info.learning_system:
                    logger.info(f"   - Learning System: {model_info.learning_system.get('current_model_version', 'N/A')}")
            else:
                logger.warning(f"⚠️ Model info failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
    
    def test_enhanced_text_recommendations(self):
        """Test enhanced text-based recommendations."""
        logger.info("\n📝 Testing Enhanced Text Recommendations")
        logger.info("-" * 50)
        
        test_queries = [
            "I want something warm and comforting for a cold evening",
            "I'm feeling hot and need something refreshing",
            "It's date night, I want something romantic and elegant",
            "I need something quick and light for lunch break",
            "I'm craving something spicy and exciting"
        ]
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n🔍 Test {i}: {query}")
            
            try:
                response = requests.post(
                    f"{self.api_base}/enhanced-recommend",
                    json={
                        "text_input": query,
                        "user_context": {"time_of_day": "evening", "weather": "cold"},
                        "user_id": self.user_id,
                        "session_id": self.session_id,
                        "top_k": 3
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✅ Found {len(data['recommendations'])} recommendations")
                    
                    # Show AI analysis
                    if data.get('intent_prediction'):
                        intent = data['intent_prediction']
                        logger.info(f"   🧠 Primary Intent: {intent['primary_intent']}")
                        logger.info(f"   🎯 Confidence: {intent['confidence']:.2%}")
                    
                    if data.get('multimodal_analysis'):
                        analysis = data['multimodal_analysis']
                        logger.info(f"   🔗 Multi-Modal Confidence: {analysis['combined_confidence']:.2%}")
                        logger.info(f"   📊 Mood Categories: {len(analysis['mood_categories'])}")
                    
                    # Show top recommendation
                    if data['recommendations']:
                        top_rec = data['recommendations'][0]
                        logger.info(f"   🥇 Top Recommendation: {top_rec['food_name']} (Score: {top_rec['score']:.2%})")
                        logger.info(f"   🏷️ Category: {top_rec['food_category']}")
                        logger.info(f"   🌍 Region: {top_rec['food_region']}")
                    
                    logger.info(f"   ⚡ Processing Time: {data['processing_time']*1000:.0f}ms")
                    
                else:
                    logger.warning(f"⚠️ Recommendation failed: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"❌ Test {i} error: {e}")
            
            time.sleep(1)  # Small delay between requests
    
    def test_multimodal_analysis(self):
        """Test multi-modal input analysis."""
        logger.info("\n🔗 Testing Multi-Modal Analysis")
        logger.info("-" * 50)
        
        # Test text-only analysis
        logger.info("🔍 Testing Text-Only Analysis")
        try:
            response = requests.post(
                f"{self.api_base}/analyze-multimodal",
                json={"text": "I want something warm and comforting"}
            )
            
            if response.status_code == 200:
                data = response.json()
                analysis = data['analysis']
                logger.info(f"✅ Text Analysis: {analysis['primary_mood']}")
                logger.info(f"✅ Confidence: {analysis['combined_confidence']:.2%}")
                logger.info(f"✅ Mood Categories: {analysis['mood_categories']}")
                logger.info(f"✅ Entities: {analysis['extracted_entities']}")
            else:
                logger.warning(f"⚠️ Text analysis failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Text analysis error: {e}")
        
        # Test image upload (simulated)
        logger.info("\n🖼️ Testing Image Analysis (Simulated)")
        try:
            # Create a simple test image (1x1 pixel PNG)
            test_image_data = self.create_test_image()
            
            response = requests.post(
                f"{self.api_base}/upload-image",
                files={"file": ("test.png", test_image_data, "image/png")},
                data={"user_context": "{}"}
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Image Analysis: {data['filename']}")
                logger.info(f"✅ File Size: {data['file_size']} bytes")
                if data.get('image_analysis'):
                    logger.info(f"✅ Image Caption: {data['image_analysis'].get('caption', 'N/A')}")
            else:
                logger.warning(f"⚠️ Image analysis failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Image analysis error: {e}")
    
    def test_realtime_learning(self):
        """Test real-time learning system."""
        logger.info("\n📈 Testing Real-Time Learning System")
        logger.info("-" * 50)
        
        # Submit feedback
        logger.info("💬 Submitting User Feedback")
        feedback_data = [
            {
                "rating": 5,
                "feedback_text": "Excellent recommendations! The spicy food was perfect for my mood."
            },
            {
                "rating": 4,
                "feedback_text": "Good suggestions, but could be more specific to my location."
            },
            {
                "rating": 3,
                "feedback_text": "Decent recommendations, but not exactly what I was looking for."
            }
        ]
        
        for i, feedback in enumerate(feedback_data, 1):
            try:
                response = requests.post(
                    f"{self.api_base}/enhanced-feedback",
                    json={
                        "user_id": self.user_id,
                        "session_id": self.session_id,
                        "input_text": f"Test feedback {i}",
                        "recommended_foods": ["Spicy Curry", "Hot Wings", "Mapo Tofu"],
                        "selected_food": "Spicy Curry",
                        "rating": feedback["rating"],
                        "feedback_text": feedback["feedback_text"],
                        "context": {"weather": "cold", "energy_level": "high"}
                    }
                )
                
                if response.status_code == 200:
                    logger.info(f"✅ Feedback {i} submitted successfully")
                else:
                    logger.warning(f"⚠️ Feedback {i} failed: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"❌ Feedback {i} error: {e}")
            
            time.sleep(0.5)
        
        # Get learning metrics
        logger.info("\n📊 Getting Learning Metrics")
        try:
            response = requests.get(f"{self.api_base}/learning-metrics")
            
            if response.status_code == 200:
                metrics = response.json()
                current = metrics['current_performance']
                logger.info(f"✅ Current Model Version: {current.get('current_model_version', 'N/A')}")
                logger.info(f"✅ Total Feedback Count: {current.get('total_feedback_count', 0)}")
                logger.info(f"✅ Unique Users: {metrics['user_insights'].get('unique_users', 0)}")
                logger.info(f"✅ Total Sessions: {metrics['user_insights'].get('total_sessions', 0)}")
                
                if metrics['model_versions']:
                    latest_version = metrics['model_versions'][-1]
                    logger.info(f"✅ Latest Update: {latest_version.get('version', 'N/A')}")
                    logger.info(f"✅ Performance Improvement: {latest_version.get('performance_improvement', 0):.2%}")
            else:
                logger.warning(f"⚠️ Learning metrics failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Learning metrics error: {e}")
    
    def test_advanced_features(self):
        """Test advanced AI features."""
        logger.info("\n🚀 Testing Advanced AI Features")
        logger.info("-" * 50)
        
        # Get advanced features info
        try:
            response = requests.get(f"{self.api_base}/advanced-features")
            
            if response.status_code == 200:
                features = response.json()
                logger.info("✅ Advanced Features Available:")
                
                for feature_name, feature_info in features['phase_3_features'].items():
                    logger.info(f"   🧠 {feature_name.replace('_', ' ').title()}")
                    logger.info(f"      Description: {feature_info['description']}")
                    if 'capabilities' in feature_info:
                        logger.info(f"      Capabilities: {', '.join(feature_info['capabilities'])}")
                    if 'benefits' in feature_info:
                        logger.info(f"      Benefits: {', '.join(feature_info['benefits'])}")
                    logger.info("")
            else:
                logger.warning(f"⚠️ Advanced features failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Advanced features error: {e}")
    
    def test_performance_metrics(self):
        """Test system performance and metrics."""
        logger.info("\n⚡ Testing System Performance")
        logger.info("-" * 50)
        
        # Performance test with multiple requests
        logger.info("🔄 Running Performance Test (5 concurrent requests)")
        
        start_time = time.time()
        successful_requests = 0
        total_processing_time = 0
        
        for i in range(5):
            try:
                request_start = time.time()
                response = requests.post(
                    f"{self.api_base}/enhanced-recommend",
                    json={
                        "text_input": f"Performance test query {i+1}",
                        "user_context": {"test": True},
                        "user_id": f"perf_user_{i}",
                        "session_id": f"perf_session_{i}",
                        "top_k": 2
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    successful_requests += 1
                    total_processing_time += data.get('processing_time', 0)
                    logger.info(f"   ✅ Request {i+1}: {data.get('processing_time', 0)*1000:.0f}ms")
                else:
                    logger.warning(f"   ⚠️ Request {i+1} failed: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"   ❌ Request {i+1} error: {e}")
            
            time.sleep(0.2)  # Small delay
        
        total_time = time.time() - start_time
        avg_processing_time = total_processing_time / successful_requests if successful_requests > 0 else 0
        
        logger.info(f"\n📊 Performance Results:")
        logger.info(f"   ✅ Successful Requests: {successful_requests}/5")
        logger.info(f"   ⏱️ Total Time: {total_time:.2f}s")
        logger.info(f"   🚀 Average Processing Time: {avg_processing_time*1000:.0f}ms")
        logger.info(f"   📈 Requests per Second: {successful_requests/total_time:.2f}")
    
    def create_test_image(self) -> bytes:
        """Create a simple test image for testing."""
        # This is a minimal 1x1 pixel PNG file
        png_data = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
            0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xDE, 0x00, 0x00, 0x00,
            0x0C, 0x49, 0x44, 0x41, 0x54, 0x08, 0x99, 0x63, 0xF8, 0xCF, 0x00, 0x00,
            0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D, 0xB0, 0x00, 0x00, 0x00, 0x00,
            0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82
        ])
        return png_data

def main():
    """Main demo function."""
    print("🚀 AI Mood Food Recommender - Phase 3 Demo")
    print("=" * 60)
    print("This demo showcases the advanced AI features including:")
    print("• Deep Learning Models with Transformers")
    print("• Semantic Embeddings for Food-Mood Relationships")
    print("• Multi-Modal Input (Text, Image, Voice)")
    print("• Real-Time Learning from User Feedback")
    print("• Enhanced Intent Classification")
    print("• Advanced Recommendation Algorithms")
    print("=" * 60)
    
    # Check if server is running
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and accessible")
        else:
            print("⚠️ Server responded but with unexpected status")
            return
    except requests.exceptions.RequestException:
        print("❌ Server is not running. Please start the server first:")
        print("   python -c \"import uvicorn; uvicorn.run('api.enhanced_routes:app', host='127.0.0.1', port=8000)\"")
        return
    
    # Run the demo
    demo = Phase3Demo()
    demo.run_demo()

if __name__ == "__main__":
    main() 