# AI Mood-Based Food Recommendation System 🍽️🧠

An innovative AI-powered food recommendation system that understands your eating mood and situational needs expressed in natural language, moving beyond traditional cuisine-based search to provide truly personalized food suggestions.

## 🌟 What Makes This System Special?

Instead of browsing rigid categories like "Pizza" or "Sushi," users describe their eating mood and situation:

- **"I want something light"** → Smoothies, salads, light soups
- **"I'm feeling cold and want something warm"** → Hot pot, ramen, beef stew
- **"I want something romantic for a couple's date"** → Oysters, chocolate fondue, wine pairings
- **"I'm craving something sweet/spicy/salty"** → Desserts, spicy dishes, savory snacks
- **"I want something comforting because I'm tired/sick"** → Chicken soup, mac and cheese, pudding

## 🏗️ System Architecture

### Phase 1: Comprehensive Mood-Based Taxonomy ✅
- **Weather-based**: Hot/cold weather food preferences
- **Energy-based**: Light/heavy/greasy food choices
- **Emotional**: Comfort, romantic, celebratory food
- **Flavor profile**: Sweet, spicy, salty, umami preferences
- **Occasion-based**: Family dinner, lunch break, party snacks
- **Health-based**: Recovery, detox, gentle digestion

### Phase 2: AI-Powered Recommendation Engine ✅
- **Natural Language Understanding**: Intent classification and mood extraction
- **Context Awareness**: Time, weather, location, social context
- **Smart Matching**: Mood-to-food mapping with vector similarity
- **Restaurant Integration**: Rating, reviews, delivery, proximity
- **Personalization**: User preferences and feedback learning
- **Ranking & Clustering**: Multi-factor scoring and result organization

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the System
```bash
python run.py
```

### 3. Test the System
```bash
# Quick test
python demo_system.py --quick

# Comprehensive demo
python demo_system.py
```

## 📡 API Endpoints

### Core Endpoints

#### `POST /recommend` - Get Food Recommendations
```json
{
  "user_input": "I want something warm and comforting",
  "user_context": {
    "weather": "cold",
    "time_of_day": "evening",
    "social_context": "alone"
  },
  "user_id": "user123",
  "top_k": 10
}
```

#### `POST /analyze-mood` - Analyze User Intent
```json
{
  "user_input": "I'm feeling hot and need something refreshing",
  "context": {
    "weather": "hot",
    "time_of_day": "afternoon"
  }
}
```

#### `GET /taxonomy` - View Mood Categories
Returns all available mood categories with descriptors and example foods.

#### `GET /examples` - Get Example Queries
Provides sample queries for each mood category to help users understand the system.

### Personalization Endpoints

#### `POST /preferences` - Update User Preferences
```json
{
  "user_id": "user123",
  "preferences": {
    "preferred_categories": ["EMOTIONAL_COMFORT", "FLAVOR_SPICY"],
    "preferred_tags": ["comforting", "spicy", "warm"],
    "preferred_cultures": ["Italian", "Indian"],
    "price_range": "$$",
    "delivery_preferred": true
  }
}
```

#### `POST /feedback` - Submit Feedback
```json
{
  "user_id": "user123",
  "food_item_id": "curry_001",
  "rating": 4.5,
  "feedback": "Excellent recommendation! Perfect for my mood."
}
```

## 🧠 How It Works

### 1. Intent Understanding
The system analyzes natural language input to extract:
- **Primary mood**: Main emotional or situational need
- **Secondary moods**: Additional context and preferences
- **Entities**: Weather, time, social context, health status

### 2. Mood Mapping
Uses vector similarity to map user intent to predefined mood categories:
- Creates mood vectors from taxonomy labels
- Combines primary, secondary, and context vectors
- Finds best-matching food categories

### 3. Food Matching
Matches mood categories to food items using:
- **Direct category matching**: Exact mood category alignment
- **Label matching**: Semantic label similarity
- **Tag matching**: Characteristic tag alignment
- **Context matching**: Time, weather, social appropriateness

### 4. Restaurant Integration
Finds suitable restaurants by:
- **Mood category alignment**: Restaurant mood compatibility
- **Context matching**: Price, delivery, location preferences
- **Quality metrics**: Ratings, reviews, popularity

### 5. Ranking & Personalization
Final scoring combines:
- **Mood match score** (40%): How well food fits the mood
- **Context match score** (30%): Appropriateness for situation
- **Restaurant quality** (20%): Ratings, reviews, features
- **Personalization** (10%): User preferences and history

## 📊 Example Queries & Responses

### Query: "I'm feeling hot and need something refreshing"
**Mood Analysis:**
- Primary Mood: WEATHER_HOT
- Categories: ["weather_hot", "sensory_refreshing", "sensory_cooling"]
- Confidence: 0.9

**Recommendations:**
1. **Ice Cream** (Score: 0.95)
   - Restaurant: Fresh & Light (4.7/5)
   - Reasoning: Matches your mood: Weather Hot; Characteristics: Cold, Sweet, Refreshing; Great for hot weather

2. **Smoothie** (Score: 0.92)
   - Restaurant: Fresh & Light (4.7/5)
   - Reasoning: Matches your mood: Weather Hot; Characteristics: Cold, Healthy, Refreshing

### Query: "I want something comforting because I'm stressed"
**Mood Analysis:**
- Primary Mood: EMOTIONAL_COMFORT
- Categories: ["emotion_comfort", "goal_soothing", "sensory_warm"]
- Confidence: 0.85

**Recommendations:**
1. **Mac and Cheese** (Score: 0.93)
   - Restaurant: Comfort Kitchen (4.5/5)
   - Reasoning: Matches your mood: Emotional Comfort; Characteristics: Comforting, Cheesy, Warm; Perfect for evening

2. **Chicken Soup** (Score: 0.91)
   - Restaurant: Comfort Kitchen (4.5/5)
   - Reasoning: Matches your mood: Emotional Comfort; Characteristics: Comforting, Warming, Nurturing

## 🔧 Technical Implementation

### Core Components

#### `MoodMapper` (`core/mood_mapper.py`)
- Loads and processes mood-food taxonomy
- Creates mood vectors for similarity matching
- Extracts entities and context from user input
- Maps moods to food recommendations

#### `MoodBasedRecommendationEngine` (`core/recommendation/recommendation_algorithm.py`)
- Orchestrates the complete recommendation process
- Analyzes user intent and context
- Finds mood-food and mood-restaurant matches
- Combines and ranks final recommendations
- Handles personalization and feedback

#### `API Routes` (`api/routes.py`)
- RESTful API endpoints for all system functions
- Request/response validation with Pydantic
- Error handling and status codes
- CORS support for web applications

### Data Structures

#### Food Items
```python
@dataclass
class FoodItem:
    name: str
    category: str
    region: str
    culture: str
    tags: List[str]
    labels: List[str]
    descriptors: List[str]
    nutrition_score: Optional[float]
    popularity_score: Optional[float]
```

#### Restaurants
```python
@dataclass
class Restaurant:
    id: str
    name: str
    cuisine_type: str
    rating: float
    review_count: int
    price_range: str
    location: Dict[str, float]
    delivery_available: bool
    tags: List[str]
    mood_categories: List[str]
```

#### Recommendations
```python
@dataclass
class Recommendation:
    food_item: FoodItem
    restaurant: Optional[Restaurant]
    score: float
    reasoning: List[str]
    mood_match: float
    context_match: float
    personalization_score: float
```

## 🎯 Use Cases

### 1. **Personal Food Discovery**
- Find foods that match your current mood
- Discover new dishes based on emotional needs
- Get contextually appropriate suggestions

### 2. **Restaurant Recommendations**
- Find restaurants that serve mood-appropriate food
- Consider ratings, delivery, and proximity
- Match cuisine types to emotional preferences

### 3. **Health & Wellness**
- Gentle foods for recovery and illness
- Detox and clean eating options
- Comfort foods for emotional support

### 4. **Social Dining**
- Romantic dinner suggestions
- Family-friendly meal options
- Party and celebration food ideas

### 5. **Weather & Seasonal**
- Hot weather refreshments
- Cold weather warming foods
- Seasonal appropriateness

## 🚀 Future Enhancements

### Phase 3: Advanced AI Features
- **Deep Learning Models**: Enhanced intent classification with transformers
- **Semantic Embeddings**: Better understanding of food-mood relationships
- **Multi-modal Input**: Image, voice, and text input support
- **Real-time Learning**: Continuous improvement from user feedback

### Phase 4: Enterprise Features
- **Multi-language Support**: Global cuisine and cultural preferences
- **Advanced Analytics**: User behavior insights and trends
- **Integration APIs**: Third-party restaurant and delivery platforms
- **Mobile Applications**: Native iOS and Android apps

## 🤝 Contributing

We welcome contributions! Areas for improvement:
- Additional mood categories and food mappings
- Enhanced NLU models and intent classification
- More sophisticated recommendation algorithms
- Additional restaurant data sources
- User interface improvements

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Hugging Face Transformers**: For NLU capabilities
- **FastAPI**: For high-performance API framework
- **Pydantic**: For data validation and serialization
- **Scikit-learn**: For machine learning utilities

---

**Built with ❤️ for food lovers who want their meals to match their mood!**

*"Tell me how you feel, and I'll tell you what to eat!"* 🍽️✨
