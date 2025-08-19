import logging
from typing import List, Set
from utils.label_utils import normalize_label_name

logger = logging.getLogger(__name__)

class LabelValidator:
    def __init__(self, taxonomy_labels: List[str]):
        """Initialize validator with taxonomy labels."""
        self.taxonomy_labels = set(taxonomy_labels)
        self.normalized_taxonomy = {normalize_label_name(label) for label in taxonomy_labels}
        logger.info(f"Initialized validator with {len(self.taxonomy_labels)} taxonomy labels")
    
    def validate_labels(self, labels: List[str]) -> List[str]:
        """Validate and normalize labels against taxonomy."""
        if not labels:
            return []
        
        logger.info(f"Validating {len(labels)} labels: {labels}")
        
        # Step 1: Normalize all labels
        normalized_labels = []
        for label in labels:
            normalized = normalize_label_name(label)
            normalized_labels.append(normalized)
            logger.debug(f"Normalized '{label}' -> '{normalized}'")
        
        # Step 2: Filter labels that exist in taxonomy
        valid_labels = []
        for original, normalized in zip(labels, normalized_labels):
            if normalized in self.normalized_taxonomy:
                # Find the original taxonomy label (preserve exact case/format)
                for taxonomy_label in self.taxonomy_labels:
                    if normalize_label_name(taxonomy_label) == normalized:
                        valid_labels.append(taxonomy_label)
                        break
                logger.debug(f"Validated '{original}' -> '{taxonomy_label}'")
            else:
                logger.warning(f"Invalid label '{original}' (normalized: '{normalized}') not in taxonomy")
        
        # Step 3: Remove duplicates while preserving order
        seen = set()
        unique_labels = []
        for label in valid_labels:
            if label not in seen:
                seen.add(label)
                unique_labels.append(label)
        
        logger.info(f"Validation result: {len(unique_labels)} valid labels: {unique_labels}")
        return unique_labels
    
    def get_invalid_labels(self, labels: List[str]) -> List[str]:
        """Get list of labels that are not in taxonomy."""
        if not labels:
            return []
        
        invalid = []
        for label in labels:
            normalized = normalize_label_name(label)
            if normalized not in self.normalized_taxonomy:
                invalid.append(label)
        
        return invalid
    
    def get_suggestions(self, invalid_label: str) -> List[str]:
        """Get taxonomy suggestions for an invalid label."""
        normalized_invalid = normalize_label_name(invalid_label)
        suggestions = []
        
        # Simple fuzzy matching - could be improved with edit distance
        for taxonomy_label in self.taxonomy_labels:
            normalized_taxonomy = normalize_label_name(taxonomy_label)
            if (normalized_invalid in normalized_taxonomy or 
                normalized_taxonomy in normalized_invalid):
                suggestions.append(taxonomy_label)
        
        return suggestions[:5]  # Limit to 5 suggestions

def validate_labels(labels: List[str], taxonomy_labels: List[str]) -> List[str]:
    """Convenience function to validate labels against taxonomy."""
    validator = LabelValidator(taxonomy_labels)
    return validator.validate_labels(labels)
