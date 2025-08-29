#!/usr/bin/env python3
"""
Extend Mood Food Taxonomy Script
================================

This script extends the mood_food_taxonomy.json from 20 to 138 categories
to align with the ML intent classifier expectations.
"""

import json
import random
from pathlib import Path

def load_unified_labels():
    """Load the unified label mappings from the ML model."""
    label_file = Path("models/intent_classifier/unified_label_mappings.json")
    
    if not label_file.exists():
        print("❌ Unified label mappings file not found!")
        return []
    
    with open(label_file, 'r') as f:
        data = json.load(f)
    
    # Extract all 138 labels
    labels = []
    for i in range(138):
        label = data["unified_id_to_label"].get(str(i))
        if label:
            labels.append(label)
    
    print(f"✅ Loaded {len(labels)} unified labels from ML model")
    return labels

def load_current_taxonomy():
    """Load the current taxonomy file."""
    taxonomy_file = Path("data/taxonomy/mood_food_taxonomy.json")
    
    if not taxonomy_file.exists():
        print("❌ Current taxonomy file not found!")
        return {}
    
    with open(taxonomy_file, 'r') as f:
        taxonomy = json.load(f)
    
    print(f"✅ Loaded current taxonomy with {len(taxonomy)} categories")
    return taxonomy

def generate_category_data(label):
    """Generate taxonomy data for a new category based on the label."""
    
    # Define food templates for different label types
    food_templates = {
        # Emotion-based categories
        "emotion_": {
            "descriptors": ["emotional", "mood-based", "feeling", "psychological"],
            "foods": ["comfort food", "mood-lifting dish", "emotional support meal"]
        },
        # Goal-based categories
        "goal_": {
            "descriptors": ["purpose-driven", "intentional", "targeted", "specific"],
            "foods": ["goal-oriented dish", "purposeful meal", "targeted nutrition"]
        },
        # Health-based categories
        "health_": {
            "descriptors": ["health-conscious", "wellness", "nourishing", "beneficial"],
            "foods": ["healthy option", "wellness meal", "nourishing dish"]
        },
        # Sensory categories
        "sensory_": {
            "descriptors": ["sensory experience", "texture-focused", "flavor-driven"],
            "foods": ["sensory-rich dish", "texture-focused meal", "flavor experience"]
        },
        # Time-based categories
        "time_": {
            "descriptors": ["time-specific", "moment-appropriate", "timing-focused"],
            "foods": ["time-appropriate meal", "moment-specific dish", "timing-focused food"]
        },
        # Weather categories
        "weather_": {
            "descriptors": ["weather-appropriate", "climate-considerate", "seasonal"],
            "foods": ["weather-suitable dish", "climate-appropriate meal", "seasonal food"]
        },
        # Social categories
        "social_": {
            "descriptors": ["social context", "interpersonal", "communal"],
            "foods": ["social meal", "communal dish", "interpersonal food"]
        },
        # Cuisine categories
        "cuisine_": {
            "descriptors": ["cultural", "regional", "traditional", "authentic"],
            "foods": ["cultural dish", "regional specialty", "traditional meal"]
        },
        # Activity categories
        "activity_": {
            "descriptors": ["activity-focused", "performance-oriented", "energetic"],
            "foods": ["activity-appropriate meal", "performance food", "energetic dish"]
        },
        # Atmosphere categories
        "atmosphere_": {
            "descriptors": ["ambiance-focused", "environment-considerate", "setting-appropriate"],
            "foods": ["ambiance-appropriate dish", "environment-suitable meal", "setting-focused food"]
        }
    }
    
    # Find the appropriate template
    template = None
    for prefix, data in food_templates.items():
        if label.startswith(prefix):
            template = data
            break
    
    if not template:
        # Default template for unknown label types
        template = {
            "descriptors": ["general", "versatile", "adaptable"],
            "foods": ["versatile dish", "adaptable meal", "general food"]
        }
    
    # Generate specific descriptors and foods based on the label
    base_descriptors = template["descriptors"]
    base_foods = template["foods"]
    
    # Add label-specific descriptors
    specific_descriptors = [label.replace("_", " "), f"{label}_focused", f"{label}_oriented"]
    
    # Generate realistic food names based on the label
    food_categories = {
        "breakfast": ["oatmeal", "eggs benedict", "pancakes", "yogurt parfait", "smoothie bowl"],
        "lunch": ["quinoa salad", "turkey sandwich", "soup", "wrap", "bento box"],
        "dinner": ["grilled salmon", "pasta carbonara", "beef stew", "curry", "roasted chicken"],
        "dessert": ["chocolate cake", "tiramisu", "ice cream", "fruit tart", "creme brulee"],
        "snack": ["nuts", "chips", "popcorn", "cheese", "fruit"],
        "drink": ["tea", "coffee", "smoothie", "juice", "water"]
    }
    
    # Select appropriate food category based on label
    if "breakfast" in label or "morning" in label:
        foods = food_categories["breakfast"]
    elif "lunch" in label or "midday" in label:
        foods = food_categories["lunch"]
    elif "dinner" in label or "evening" in label:
        foods = food_categories["dinner"]
    elif "dessert" in label or "sweet" in label:
        foods = food_categories["dessert"]
    elif "snack" in label:
        foods = food_categories["snack"]
    elif "drink" in label or "beverage" in label:
        foods = food_categories["drink"]
    else:
        foods = food_categories["lunch"]  # Default to lunch
    
    # Create the category data
    category_data = {
        "descriptors": base_descriptors + specific_descriptors,
        "labels": [label],
        "foods": []
    }
    
    # Generate 3-5 food items with detailed information
    num_foods = random.randint(3, 5)
    selected_foods = random.sample(foods, min(num_foods, len(foods)))
    
    for food in selected_foods:
        food_item = {
            "name": food,
            "region": "Global",
            "culture": "Universal",
            "tags": [label.replace("_", " "), "versatile", "adaptable"]
        }
        category_data["foods"].append(food_item)
    
    return category_data

def extend_taxonomy():
    """Extend the taxonomy from 20 to 138 categories."""
    
    # Load current taxonomy and unified labels
    current_taxonomy = load_current_taxonomy()
    unified_labels = load_unified_labels()
    
    if not unified_labels:
        return False
    
    # Get existing category names
    existing_categories = set(current_taxonomy.keys())
    
    # Create new categories for missing labels
    new_categories = {}
    added_count = 0
    
    for label in unified_labels:
        # Convert label to category name format
        category_name = label.upper()
        
        # Skip if category already exists
        if category_name in existing_categories:
            continue
        
        # Generate new category data
        category_data = generate_category_data(label)
        new_categories[category_name] = category_data
        added_count += 1
    
    # Merge new categories with existing ones
    extended_taxonomy = {**current_taxonomy, **new_categories}
    
    print(f"✅ Generated {added_count} new categories")
    print(f"✅ Total categories: {len(extended_taxonomy)}")
    
    # Save the extended taxonomy
    output_file = Path("data/taxonomy/mood_food_taxonomy.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(extended_taxonomy, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Extended taxonomy saved to {output_file}")
    
    return extended_taxonomy

def validate_taxonomy(taxonomy):
    """Validate the extended taxonomy."""
    
    # Check total count
    total_categories = len(taxonomy)
    print(f"\n🔍 Validation Results:")
    print(f"   Total categories: {total_categories}")
    
    if total_categories == 138:
        print("   ✅ Category count matches ML model expectation (138)")
    else:
        print(f"   ❌ Category count mismatch: expected 138, got {total_categories}")
        return False
    
    # Check structure consistency
    structure_errors = 0
    for category_name, category_data in taxonomy.items():
        required_fields = ["descriptors", "labels", "foods"]
        for field in required_fields:
            if field not in category_data:
                print(f"   ❌ Missing field '{field}' in category '{category_name}'")
                structure_errors += 1
        
        if "foods" in category_data and not isinstance(category_data["foods"], list):
            print(f"   ❌ Invalid foods field in category '{category_name}'")
            structure_errors += 1
    
    if structure_errors == 0:
        print("   ✅ All categories have consistent structure")
    else:
        print(f"   ❌ Found {structure_errors} structure errors")
        return False
    
    return True

def show_sample_categories(taxonomy, count=5):
    """Show a sample of random categories for manual review."""
    
    print(f"\n📋 Sample Categories (showing {count} random examples):")
    print("=" * 60)
    
    # Get random categories (excluding the original 20)
    original_categories = {
        "WEATHER_HOT", "WEATHER_COLD", "ENERGY_LIGHT", "ENERGY_HEAVY", "ENERGY_GREASY",
        "EMOTIONAL_COMFORT", "EMOTIONAL_ROMANTIC", "EMOTIONAL_CELEBRATORY",
        "FLAVOR_SWEET", "FLAVOR_SPICY", "FLAVOR_SALTY", "OCCASION_FAMILY_DINNER",
        "OCCASION_LUNCH_BREAK", "OCCASION_HANGOVER_CURE", "OCCASION_PARTY_SNACKS",
        "HEALTH_ILLNESS", "HEALTH_DETOX", "GOAL_COMFORT", "SENSORY_WARMING", "SEASON_WINTER"
    }
    
    new_categories = [cat for cat in taxonomy.keys() if cat not in original_categories]
    sample_categories = random.sample(new_categories, min(count, len(new_categories)))
    
    for i, category_name in enumerate(sample_categories, 1):
        category_data = taxonomy[category_name]
        
        print(f"\n{i}. {category_name}")
        print(f"   Descriptors: {', '.join(category_data['descriptors'][:3])}...")
        print(f"   Labels: {', '.join(category_data['labels'])}")
        print(f"   Foods: {len(category_data['foods'])} items")
        
        # Show first food item as example
        if category_data['foods']:
            first_food = category_data['foods'][0]
            print(f"   Example: {first_food['name']} ({first_food['region']}, {first_food['culture']})")

def main():
    """Main function to extend the taxonomy."""
    
    print("🚀 Extending Mood Food Taxonomy from 20 → 138 categories")
    print("=" * 60)
    
    # Extend the taxonomy
    extended_taxonomy = extend_taxonomy()
    
    if not extended_taxonomy:
        print("❌ Failed to extend taxonomy")
        return
    
    # Validate the extended taxonomy
    if not validate_taxonomy(extended_taxonomy):
        print("❌ Taxonomy validation failed")
        return
    
    # Show sample categories
    show_sample_categories(extended_taxonomy)
    
    print(f"\n🎉 Taxonomy extension completed successfully!")
    print(f"   Original categories: 20")
    print(f"   New categories: {len(extended_taxonomy) - 20}")
    print(f"   Total categories: {len(extended_taxonomy)}")
    print(f"   File: data/taxonomy/mood_food_taxonomy.json")

if __name__ == "__main__":
    main()
