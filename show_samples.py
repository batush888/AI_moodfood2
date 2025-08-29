#!/usr/bin/env python3
import json

# Load the extended taxonomy
with open('data/taxonomy/mood_food_taxonomy.json', 'r') as f:
    taxonomy = json.load(f)

# Show 5 more random categories
print('📋 Additional Sample Categories:')
print('=' * 50)

# Get some interesting new categories
interesting_categories = ['CUISINE_ASIAN', 'EMOTION_ADVENTUROUS', 'GOAL_ENERGY', 'SENSORY_FRESH', 'SOCIAL_FRIENDS']
for i, cat_name in enumerate(interesting_categories, 1):
    if cat_name in taxonomy:
        cat_data = taxonomy[cat_name]
        print(f'\n{i}. {cat_name}')
        print(f'   Descriptors: {", ".join(cat_data["descriptors"][:4])}...')
        print(f'   Labels: {", ".join(cat_data["labels"])}')
        print(f'   Foods: {len(cat_data["foods"])} items')
        if cat_data['foods']:
            first_food = cat_data['foods'][0]
            print(f'   Example: {first_food["name"]} ({first_food["region"]}, {first_food["culture"]})')
            print(f'   Tags: {", ".join(first_food["tags"])}')

print(f'\n✅ Total categories in taxonomy: {len(taxonomy)}')
