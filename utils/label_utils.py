# AI_moodfood2/utils/label_utils.py
# SINGLE SOURCE OF TRUTH FOR ALL LABELS

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Union, Set, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------
# File paths
# ------------------
BASE_DIR = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = BASE_DIR / "data" / "taxonomy" / "mood_food_taxonomy.json"
DATASET_PATH = BASE_DIR / "data" / "intent_dataset.json"

# ------------------
# Label Categories
# ------------------
class LabelCategories:
    """Centralized label categories for the AI system."""
    
    # Mood-based labels
    MOOD_LABELS = {
        'comfort', 'energy', 'celebration', 'stress', 'romance', 
        'focus', 'relaxation', 'motivation', 'creativity', 'social'
    }
    
    # Goal-based labels
    GOAL_LABELS = {
        'comfort', 'health', 'quick', 'gourmet', 'budget', 
        'dietary', 'cultural', 'seasonal', 'occasional'
    }
    
    # Emotion-based labels
    EMOTION_LABELS = {
        'happy', 'sad', 'excited', 'anxious', 'confident', 
        'tired', 'energetic', 'stressed', 'relaxed', 'focused'
    }
    
    # Cuisine type labels
    CUISINE_LABELS = {
        'italian', 'chinese', 'japanese', 'indian', 'mexican',
        'french', 'thai', 'mediterranean', 'american', 'fusion'
    }
    
    # Dietary restriction labels
    DIETARY_LABELS = {
        'vegetarian', 'vegan', 'gluten_free', 'dairy_free', 
        'low_carb', 'keto', 'paleo', 'halal', 'kosher'
    }

# ------------------
# Loaders with validation
# ------------------
def normalize_label_name(label: str) -> str:
    """Standardize label naming (lowercase + underscores)."""
    return label.strip().lower().replace(" ", "_")

def normalize_all_labels(labels):
    """Normalize a list of labels."""
    return [normalize_label_name(l) for l in labels]

def load_taxonomy(path: Path = TAXONOMY_PATH) -> Dict[str, Dict]:
    """
    Loads the mood-food taxonomy from JSON with validation and auto-normalization.
    Normalizes taxonomy labels to lowercase with underscores to match dataset format.
    """
    try:
        if not path.exists():
            logger.warning(f"Taxonomy file not found: {path}")
            return {}
        
        with open(path, "r", encoding="utf-8") as f:
            taxonomy = json.load(f)
        
        # Auto-normalize taxonomy labels to match dataset format
        normalized = {}
        for label, values in taxonomy.items():
            normalized_label = normalize_label_name(label)
            normalized[normalized_label] = values
            if normalized_label != label:
                logger.info(f"Normalized taxonomy label: '{label}' → '{normalized_label}'")
        
        logger.info(f"Loaded and normalized taxonomy with {len(normalized)} categories")
        return normalized
    except Exception as e:
        logger.error(f"Failed to load taxonomy: {e}")
        return {}

def load_dataset_labels(path: Path = DATASET_PATH) -> List[str]:
    """
    Extract all unique labels from intent_dataset.json.
    Reusable across the project for consistent label loading.
    """
    try:
        if not path.exists():
            logger.warning(f"Dataset file not found: {path}")
            return []
        
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        labels = set()
        for sample in dataset:
            for label in sample.get("labels", []):
                labels.add(normalize_label_name(label))

        logger.info(f"Loaded {len(labels)} unique normalized labels from dataset")
        return sorted(labels)
    except Exception as e:
        logger.error(f"Failed to load dataset labels: {e}")
        return []

def load_labels_from_dataset(path: Path = DATASET_PATH) -> List[str]:
    """
    Load unique labels from the dataset file with validation.
    Alias for load_dataset_labels for backward compatibility.
    """
    return load_dataset_labels(path)

# ------------------
# Label validation and normalization
# ------------------
def validate_label(label: str, allowed_labels: Set[str]) -> bool:
    """
    Validates if a label is in the allowed set.
    """
    return label in allowed_labels

def normalize_label(label: str) -> str:
    """
    Normalizes label format (lowercase, replace spaces with underscores).
    """
    return label.lower().replace(' ', '_').replace('-', '_')

def get_standardized_labels() -> Dict[str, Set[str]]:
    """
    Returns all standardized label categories.
    """
    return {
        'mood': LabelCategories.MOOD_LABELS,
        'goal': LabelCategories.GOAL_LABELS,
        'emotion': LabelCategories.EMOTION_LABELS,
        'cuisine': LabelCategories.CUISINE_LABELS,
        'dietary': LabelCategories.DIETARY_LABELS
    }

# ------------------
# Mapping creators with validation
# ------------------
def get_label_mappings(labels: List[str]) -> Dict[str, int]:
    """
    Creates a label-to-ID mapping from a list of label names.
    """
    return {label: idx for idx, label in enumerate(labels)}

def get_label_mappings_from_taxonomy_and_dataset(taxonomy: Dict, dataset_labels: List[str]) -> Dict[str, int]:
    """
    Build label→ID mapping from taxonomy + dataset.
    Reusable across the project for consistent mapping.
    """
    taxonomy_labels = list(taxonomy.keys())
    all_labels = sorted(set(taxonomy_labels) | set(dataset_labels))
    label_to_id = {label: idx for idx, label in enumerate(all_labels)}
    return label_to_id

def get_reverse_label_mappings(label_to_id: Dict[str, int]) -> Dict[int, str]:
    """
    Creates an ID-to-label mapping from a label-to-ID dictionary.
    """
    return {v: k for k, v in label_to_id.items()}

# ------------------
# Label consistency checking
# ------------------
def check_label_consistency() -> Dict[str, List[str]]:
    """
    Checks for inconsistencies between taxonomy and dataset labels.
    """
    taxonomy_labels = set(load_taxonomy().keys())
    dataset_labels = set(load_labels_from_dataset())
    
    inconsistencies = {
        'taxonomy_only': sorted(taxonomy_labels - dataset_labels),
        'dataset_only': sorted(dataset_labels - taxonomy_labels),
        'common': sorted(taxonomy_labels & dataset_labels)
    }
    
    logger.info(f"Label consistency check: {len(inconsistencies['common'])} common, "
                f"{len(inconsistencies['taxonomy_only'])} taxonomy-only, "
                f"{len(inconsistencies['dataset_only'])} dataset-only")
    
    return inconsistencies

# ------------------
# Normalization helpers with validation
# ------------------
def normalize_labels(
    labels: Union[List[str], str],
    label_to_id: Dict[str, int],
    validate: bool = True
) -> List[int]:
    """
    Converts label names to their corresponding integer IDs with optional validation.
    """
    if isinstance(labels, str):
        labels = [labels]
    
    if validate:
        valid_labels = [label for label in labels if label in label_to_id]
        if len(valid_labels) != len(labels):
            invalid = set(labels) - set(label_to_id.keys())
            logger.warning(f"Invalid labels found: {invalid}")
        labels = valid_labels
    
    return [label_to_id[label] for label in labels]

def normalize_labels_simple(labels, label_to_id):
    """
    Convert label names → IDs (simplified version).
    Reusable across the project for basic normalization.
    """
    return [label_to_id[label] for label in labels if label in label_to_id]

def denormalize_ids(
    ids: Union[List[int], int],
    id_to_label: Dict[int, str]
) -> List[str]:
    """
    Converts label IDs back to their corresponding label names.
    """
    if isinstance(ids, int):
        ids = [ids]
    return [id_to_label[idx] for idx in ids if idx in id_to_label]

def denormalize_ids_simple(ids, id_to_label):
    """
    Convert IDs → label names (simplified version).
    Reusable across the project for basic denormalization.
    """
    return [id_to_label[idx] for idx in ids if idx in id_to_label]

# ------------------
# Preload and validate all label systems
# ------------------
def initialize_label_system() -> Dict[str, any]:
    """
    Initializes and validates the entire label system.
    """
    logger.info("Initializing label system...")
    
    # Load and validate
    taxonomy = load_taxonomy()
    dataset_labels = load_dataset_labels()  # Use the new function
    
    # Create mappings
    taxonomy_labels = sorted(taxonomy.keys())
    taxonomy_label_to_id = get_label_mappings(taxonomy_labels)
    taxonomy_id_to_label = get_reverse_label_mappings(taxonomy_label_to_id)
    
    dataset_label_to_id = get_label_mappings(dataset_labels)
    dataset_id_to_label = get_reverse_label_mappings(dataset_label_to_id)
    
    # Check consistency
    consistency_report = check_label_consistency()
    
    # Create unified label system using the new function
    unified_label_to_id = get_label_mappings_from_taxonomy_and_dataset(taxonomy, dataset_labels)
    unified_id_to_label = get_reverse_label_mappings(unified_label_to_id)
    
    # Get all labels for the unified system
    all_labels = sorted(set(taxonomy_labels) | set(dataset_labels))
    
    label_system = {
        'taxonomy': taxonomy,
        'taxonomy_labels': taxonomy_labels,
        'taxonomy_label_to_id': taxonomy_label_to_id,
        'taxonomy_id_to_label': taxonomy_id_to_label,
        'dataset_labels': dataset_labels,
        'dataset_label_to_id': dataset_label_to_id,
        'dataset_id_to_label': dataset_id_to_label,
        'unified_labels': all_labels,
        'unified_label_to_id': unified_label_to_id,
        'unified_id_to_label': unified_id_to_label,
        'consistency_report': consistency_report,
        'standardized_categories': get_standardized_labels()
    }
    
    logger.info(f"Label system initialized with {len(all_labels)} unified labels")
    return label_system

# ------------------
# Global label system instance
# ------------------
LABEL_SYSTEM = initialize_label_system()

# ------------------
# Convenience accessors
# ------------------
def get_taxonomy_labels() -> List[str]:
    """Get taxonomy labels."""
    return LABEL_SYSTEM['taxonomy_labels']

def get_dataset_labels() -> List[str]:
    """Get dataset labels."""
    return LABEL_SYSTEM['dataset_labels']

def get_unified_labels() -> List[str]:
    """Get all unified labels."""
    return LABEL_SYSTEM['unified_labels']

def get_label_id(label: str, system: str = 'unified') -> Optional[int]:
    """Get label ID from specified system."""
    mapping = LABEL_SYSTEM.get(f'{system}_label_to_id', {})
    return mapping.get(label)

def get_label_name(label_id: int, system: str = 'unified') -> Optional[str]:
    """Get label name from specified system."""
    mapping = LABEL_SYSTEM.get(f'{system}_id_to_label', {})
    return mapping.get(label_id)

# ------------------
# Convenience functions for project-wide usage
# ------------------
def get_project_labels() -> Dict[str, any]:
    """
    Get all project labels in a convenient format.
    Reusable across the project for consistent label access.
    """
    return {
        'taxonomy': load_taxonomy(),
        'dataset_labels': load_dataset_labels(),
        'unified_mapping': get_label_mappings_from_taxonomy_and_dataset(
            load_taxonomy(), 
            load_dataset_labels()
        )
    }

def quick_label_normalize(labels: List[str]) -> List[int]:
    """
    Quick normalization using the unified label system.
    Reusable across the project for simple label conversion.
    """
    unified_mapping = LABEL_SYSTEM['unified_label_to_id']
    return normalize_labels_simple(labels, unified_mapping)

def quick_label_denormalize(ids: List[int]) -> List[str]:
    """
    Quick denormalization using the unified label system.
    Reusable across the project for simple ID conversion.
    """
    unified_mapping = LABEL_SYSTEM['unified_id_to_label']
    return denormalize_ids_simple(ids, unified_mapping)

# ------------------
# Main interface
# ------------------
if __name__ == "__main__":
    print("🔍 Label System Status:")
    print(f"📂 Taxonomy labels: {len(LABEL_SYSTEM['taxonomy_labels'])}")
    print(f"📂 Dataset labels: {len(LABEL_SYSTEM['dataset_labels'])}")
    print(f"🔗 Unified labels: {len(LABEL_SYSTEM['unified_labels'])}")
    
    print("\n📊 Consistency Report:")
    for category, labels in LABEL_SYSTEM['consistency_report'].items():
        print(f"  {category}: {len(labels)} labels")
    
    print("\n🎯 Standardized Categories:")
    for category, labels in LABEL_SYSTEM['standardized_categories'].items():
        print(f"  {category}: {len(labels)} labels")
