TRAINING_EXAMPLES = [
    {
        "text": "I'm feeling really stressed and want something comforting",
        "intent": "COMFORT_EMOTIONAL",
        "entities": [
            {"entity": "INTENSITY_MODIFIERS", "value": "really", "start": 12, "end": 18},
            {"entity": "EMOTION", "value": "stressed", "start": 19, "end": 27}
        ],
        "mood_intensity": 0.9
    },
    {
        "text": "It's cold outside and I need something warm and spicy",
        "intent": "WEATHER_SEASONAL",
        "entities": [
            {"entity": "WEATHER", "value": "cold", "start": 5, "end": 9},
            {"entity": "TEXTURE_PREFERENCE", "value": "warm", "start": 43, "end": 47},
            {"entity": "FLAVOR_PROFILE", "value": "spicy", "start": 52, "end": 57}
        ],
        "mood_intensity": 0.7
    }
]