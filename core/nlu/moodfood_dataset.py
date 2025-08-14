import json
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import DistilBertTokenizerFast
import torch

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Comprehensive mapping from dataset labels to taxonomy categories
LABEL_TO_TAXONOMY = {
    # Weather and Temperature
    "weather_hot": "WEATHER_HOT",
    "weather_cold": "WEATHER_COLD", 
    "weather_rainy": "WEATHER_COLD",
    "temperature_warm": "WEATHER_COLD",
    "temperature_cold": "WEATHER_COLD",
    "season_summer": "WEATHER_HOT",
    "season_winter": "WEATHER_COLD",
    
    # Goals and Intentions
    "goal_comfort": "COMFORT_EMOTIONAL",
    "goal_energy": "ENERGY_VITALITY",
    "goal_light": "HEALTH_CLEAN",
    "goal_healthy": "HEALTH_CLEAN",
    "goal_indulgence": "INDULGENCE_TREAT",
    "goal_excitement": "EXCITEMENT_ADVENTURE",
    "goal_quick": "EFFICIENCY_QUICK",
    "goal_efficient": "EFFICIENCY_QUICK",
    "goal_filling": "SATISFACTION_FILLING",
    "goal_satisfaction": "SATISFACTION_FILLING",
    "goal_soothing": "COMFORT_EMOTIONAL",
    "goal_nurturing": "COMFORT_EMOTIONAL",
    "goal_hydration": "WEATHER_HOT",
    "goal_gentle": "HEALTH_RECOVERY",
    "goal_nourishment": "HEALTH_RECOVERY",
    "goal_detox": "HEALTH_CLEAN",
    "goal_social": "SOCIAL_GROUP",
    "goal_simple": "SIMPLICITY_EASE",
    
    # Emotions and Moods
    "emotion_sad": "COMFORT_EMOTIONAL",
    "emotion_gloomy": "COMFORT_EMOTIONAL",
    "emotion_adventurous": "EXCITEMENT_ADVENTURE",
    "emotion_comfort": "COMFORT_EMOTIONAL",
    "emotion_joy": "CELEBRATION_FESTIVE",
    
    # Sensory Experiences
    "sensory_soothing": "COMFORT_EMOTIONAL",
    "sensory_exotic": "EXCITEMENT_ADVENTURE",
    "sensory_refreshing": "WEATHER_HOT",
    "sensory_cooling": "WEATHER_HOT",
    "sensory_warming": "WEATHER_COLD",
    "sensory_hearty": "WEATHER_COLD",
    "sensory_rich": "SATISFACTION_FILLING",
    "sensory_intense": "EXCITEMENT_ADVENTURE",
    "sensory_exciting": "EXCITEMENT_ADVENTURE",
    "sensory_sweet": "INDULGENCE_TREAT",
    "sensory_mild": "HEALTH_RECOVERY",
    "sensory_fresh": "HEALTH_CLEAN",
    "sensory_subtle": "HEALTH_CLEAN",
    
    # Food Types and Categories
    "food_smoothie": "WEATHER_HOT",
    "meal_breakfast": "TIME_MORNING",
    "meal_lunch": "TIME_MIDDAY",
    "meal_dinner": "TIME_EVENING",
    "meal_snack": "TIME_FLEXIBLE",
    "meal_dessert": "INDULGENCE_TREAT",
    "meal_appetizer": "SOCIAL_GROUP",
    
    # Occasions and Social Context
    "occasion_romantic": "ROMANTIC_DINNER",
    "occasion_celebration": "CELEBRATION_FESTIVE",
    "occasion_family": "FAMILY_NURTURING",
    "occasion_work": "WORK_EFFICIENCY",
    "occasion_party": "SOCIAL_GROUP",
    "occasion_home": "COMFORT_EMOTIONAL",
    "occasion_casual": "COMFORT_EMOTIONAL",
    
    # Social Dynamics
    "social_couple": "ROMANTIC_DINNER",
    "social_family": "FAMILY_NURTURING",
    "social_group": "SOCIAL_GROUP",
    "social_friends": "SOCIAL_GROUP",
    "social_alone": "SOLITUDE_REFLECTION",
    
    # Time and Schedule
    "time_breakfast": "TIME_MORNING",
    "time_morning": "TIME_MORNING",
    "time_midday": "TIME_MIDDAY",
    "time_evening": "TIME_EVENING",
    "time_late_night": "TIME_LATE_NIGHT",
    "time_urgent": "EFFICIENCY_QUICK",
    "time_flexible": "TIME_FLEXIBLE",
    
    # Activities and Context
    "activity_gym": "ACTIVITY_ENERGY",
    "location_office": "WORK_EFFICIENCY",
    "location_home": "COMFORT_EMOTIONAL",
    
    # Health and Wellness
    "health_illness": "HEALTH_RECOVERY",
    "health_recovery": "HEALTH_RECOVERY",
    "digestion_gentle": "HEALTH_RECOVERY",
    
    # Taste and Flavor Profiles
    "taste_sweet": "INDULGENCE_TREAT",
    "taste_spicy": "EXCITEMENT_ADVENTURE",
    "taste_salty": "SATISFACTION_FILLING",
    "taste_umami": "SATISFACTION_FILLING",
    
    # Texture and Cooking Methods
    "texture_greasy": "INDULGENCE_TREAT",
    "texture_bold": "EXCITEMENT_ADVENTURE",
    "texture_clean": "HEALTH_CLEAN",
    "cooking_fried": "INDULGENCE_TREAT",
    
    # Energy and Hunger Levels
    "energy_level_low": "HEALTH_CLEAN",
    "energy_level_high": "ENERGY_VITALITY",
    "hunger_level_very_hungry": "SATISFACTION_FILLING",
    
    # Atmosphere and Environment
    "atmosphere_intimate": "ROMANTIC_DINNER",
    "atmosphere_festive": "CELEBRATION_FESTIVE",
    "atmosphere_home": "COMFORT_EMOTIONAL",
    "atmosphere_fun": "SOCIAL_GROUP",
    
    # Flavor Profiles
    "flavor_profile_strong": "EXCITEMENT_ADVENTURE"
}

class MoodFoodDataset(torch.utils.data.Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
        
        # Map labels to taxonomy categories
        self.texts = []
        self.labels = []
        for entry in raw_data:
            mapped_labels = set()
            for label in entry["labels"]:
                if label in LABEL_TO_TAXONOMY:
                    mapped_labels.add(LABEL_TO_TAXONOMY[label])
            if mapped_labels:
                self.texts.append(entry["text"])
                self.labels.append(list(mapped_labels))

        # Fit multi-label binarizer on all taxonomy categories (keys)
        all_taxonomy_labels = list(set(LABEL_TO_TAXONOMY.values()))
        self.mlb = MultiLabelBinarizer(classes=all_taxonomy_labels)
        self.labels_bin = self.mlb.fit_transform(self.labels)
        if hasattr(self.labels_bin, "toarray"):
            self.labels_bin = self.labels_bin.toarray()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=64, return_tensors='pt')
        label = torch.tensor(self.labels_bin[idx].toarray().flatten() if hasattr(self.labels_bin[idx], "toarray") else self.labels_bin[idx], dtype=torch.float)

        # Proper tensor shape adjustment
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label
        }