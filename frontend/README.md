# AI Mood Food Recommender - Hybrid LLM System

## 🚀 Hybrid Intent Classification System

This system now integrates **DeepSeek LLM** for semantic parsing while maintaining our existing **taxonomy & classifier as guardrails**.

### 🔄 New Flow: LLM → Validator → Classifier

```
User Query → DeepSeek LLM → Taxonomy Validation → ML Classifier (optional) → Final Labels
```

### 📁 System Components

#### 1. **LLM Parser** (`core/nlu/llm_parser.py`)
- **DeepSeek API Integration**: Semantic understanding of user queries
- **Strict JSON Output**: Always returns `{"labels": ["label1", "label2"]}`
- **Error Handling**: Robust fallback mechanisms
- **Configuration**: `config/llm.yaml`

#### 2. **Validator** (`core/nlu/validator.py`)
- **Taxonomy Guardrails**: Ensures all labels exist in our taxonomy
- **Normalization**: Standardizes label formatting
- **Deduplication**: Removes duplicate labels
- **Suggestions**: Provides alternatives for invalid labels

#### 3. **Hybrid Pipeline** (`scripts/hybrid_infer.py`)
- **Combined Classification**: LLM + Validation + ML comparison
- **Interactive Testing**: Command-line and interactive modes
- **Data Logging**: Auto-grows dataset for future training
- **Comparison Analysis**: Shows agreement between LLM and ML

### 🎯 Usage Examples

#### Command Line
```bash
# Single query
python scripts/hybrid_infer.py "I want something warm and spicy for dinner"

# Interactive mode
python scripts/hybrid_infer.py
```

#### API Integration
```python
from core.nlu.enhanced_intent_classifier import EnhancedIntentClassifier

classifier = EnhancedIntentClassifier(taxonomy_path, use_hybrid=True)
result = await classifier.classify_intent_hybrid("I want comfort food")
```

### 📊 Output Format

```json
{
  "primary_intent": "goal_comfort",
  "confidence": 1.0,
  "all_intents": ["goal_comfort", "sensory_warming", "meal_dinner"],
  "method": "hybrid_llm",
  "llm_labels": ["goal_comfort", "sensory_warming", "meal_dinner"],
  "validated_labels": ["goal_comfort", "sensory_warming", "meal_dinner"],
  "ml_result": {...},
  "fallback": false
}
```

### 🔧 Configuration

#### LLM Settings (`config/llm.yaml`)
```yaml
provider: deepseek
model: deepseek-chat
api_key: your-api-key
max_tokens: 256
temperature: 0.0
timeout: 30
retry_attempts: 3
```

### 📈 Dataset Growth

- **Auto-Labeled Data**: `data/auto_labeled.jsonl`
- **Passive Collection**: Every query is logged with LLM + ML results
- **Future Training**: Use collected data to retrain ML classifier
- **Quality Control**: Validation ensures data quality

### 🛡️ Safety Features

1. **Taxonomy Validation**: All LLM outputs validated against our taxonomy
2. **Fallback Mechanisms**: Multiple fallback levels if LLM fails
3. **Error Handling**: Robust error handling throughout pipeline
4. **Backward Compatibility**: Existing systems continue to work

### 🚀 Benefits

1. **Immediate Power**: DeepSeek provides semantic understanding overnight
2. **Data Growth**: Passive dataset collection for future training
3. **Cost Efficiency**: Train local model when dataset reaches 5k-10k samples
4. **Privacy**: Local validation and processing
5. **Flexibility**: Easy to switch LLM providers

### 🔄 Migration Path

1. **Phase 1**: Use hybrid system alongside existing classifier
2. **Phase 2**: Collect auto-labeled data (5k-10k samples)
3. **Phase 3**: Retrain local classifier with expanded dataset
4. **Phase 4**: Switch to local-only or hybrid based on performance

### 📝 Example Prompts

The LLM receives structured prompts like:

```
User query: "I want something warm and spicy for dinner"

Taxonomy labels (choose only from this list):
goal_comfort, goal_hydration, goal_light, season_winter, sensory_warming, 
sensory_refreshing, meal_dinner, occasion_home

Return only JSON in this exact format:
{"labels": ["label1", "label2", "label3"]}
```

### 🎯 Future Enhancements

- **Provider Switching**: Easy migration between LLM providers
- **Advanced Validation**: Fuzzy matching and label suggestions
- **Performance Metrics**: Track LLM vs ML agreement over time
- **Active Learning**: Use disagreement to identify training opportunities
