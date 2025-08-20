#!/usr/bin/env python3
"""
Hybrid filtering system combining ML-based filtering with LLM fallback
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

from config.settings import (
    MIN_CONFIDENCE_THRESHOLD, LLM_FALLBACK_RANGE,
    MIN_SAMPLE_LENGTH, MAX_SAMPLE_LENGTH,
    FILTER_STATS_FILE, is_borderline_case, is_valid_sample
)
from core.filtering.llm_validator import LLMValidator

logger = logging.getLogger(__name__)

class HybridFilter:
    """Hybrid filtering system for training data quality"""
    
    def __init__(self):
        self.llm_validator = LLMValidator()
        self.filter_stats = {
            'original_count': 0,
            'duplicates_removed': 0,
            'malformed_removed': 0,
            'low_confidence_removed': 0,
            'borderline_count': 0,
            'llm_validated': 0,
            'llm_rejected': 0,
            'final_count': 0
        }
    
    def filter_training_data(self, samples: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Filter training data using hybrid approach
        
        Args:
            samples: List of training samples with 'text', 'labels', 'confidence' keys
            
        Returns:
            Tuple[List[Dict], Dict]: Filtered samples and filter statistics
        """
        logger.info(f"Starting hybrid filtering of {len(samples)} samples")
        
        # Reset stats
        self.filter_stats = {
            'original_count': len(samples),
            'duplicates_removed': 0,
            'malformed_removed': 0,
            'low_confidence_removed': 0,
            'borderline_count': 0,
            'llm_validated': 0,
            'llm_rejected': 0,
            'final_count': 0
        }
        
        # Step 1: Remove duplicates
        unique_samples = self._remove_duplicates(samples)
        logger.info(f"Removed {self.filter_stats['duplicates_removed']} duplicates")
        
        # Step 2: Remove malformed samples
        valid_samples = self._remove_malformed(unique_samples)
        logger.info(f"Removed {self.filter_stats['malformed_removed']} malformed samples")
        
        # Step 3: Apply confidence filtering and LLM validation
        filtered_samples = self._apply_confidence_filtering(valid_samples)
        logger.info(f"Removed {self.filter_stats['low_confidence_removed']} low-confidence samples")
        logger.info(f"LLM validated {self.filter_stats['llm_validated']} borderline samples")
        logger.info(f"LLM rejected {self.filter_stats['llm_rejected']} borderline samples")
        
        # Update final count
        self.filter_stats['final_count'] = len(filtered_samples)
        
        # Log final statistics
        logger.info(f"Filtering complete: {self.filter_stats['original_count']} -> {self.filter_stats['final_count']} samples")
        logger.info(f"Filtering efficiency: {self.filter_stats['final_count'] / self.filter_stats['original_count'] * 100:.1f}%")
        
        return filtered_samples, self.filter_stats
    
    def _remove_duplicates(self, samples: List[Dict]) -> List[Dict]:
        """Remove duplicate samples based on text content"""
        seen_texts = set()
        unique_samples = []
        
        for sample in samples:
            text = sample.get('text', '').strip().lower()
            if text and text not in seen_texts:
                seen_texts.add(text)
                unique_samples.append(sample)
            else:
                self.filter_stats['duplicates_removed'] += 1
        
        return unique_samples
    
    def _remove_malformed(self, samples: List[Dict]) -> List[Dict]:
        """Remove malformed or invalid samples"""
        valid_samples = []
        
        for sample in samples:
            text = sample.get('text', '')
            confidence = sample.get('confidence', 0)
            
            if is_valid_sample(text, confidence):
                valid_samples.append(sample)
            else:
                self.filter_stats['malformed_removed'] += 1
        
        return valid_samples
    
    def _apply_confidence_filtering(self, samples: List[Dict]) -> List[Dict]:
        """Apply confidence-based filtering with LLM fallback for borderline cases"""
        filtered_samples = []
        borderline_samples = []
        
        for sample in samples:
            text = sample.get('text', '')
            labels = sample.get('labels', [])
            confidence = sample.get('confidence', 0)
            
            # High confidence samples are automatically accepted
            if confidence >= MIN_CONFIDENCE_THRESHOLD:
                filtered_samples.append(sample)
                continue
            
            # Low confidence samples are rejected
            if confidence < LLM_FALLBACK_RANGE[0]:
                self.filter_stats['low_confidence_removed'] += 1
                continue
            
            # Borderline cases go to LLM validation
            if is_borderline_case(confidence):
                self.filter_stats['borderline_count'] += 1
                borderline_samples.append({
                    'sample': sample,
                    'query': text,
                    'label': labels[0] if labels else 'unknown',
                    'confidence': confidence
                })
        
        # Process borderline cases with LLM
        if borderline_samples:
            logger.info(f"Processing {len(borderline_samples)} borderline cases with LLM")
            llm_results = self.llm_validator.validate_batch(borderline_samples)
            
            for i, (sample_data, is_valid) in enumerate(zip(borderline_samples, llm_results)):
                if is_valid:
                    filtered_samples.append(sample_data['sample'])
                    self.filter_stats['llm_validated'] += 1
                else:
                    self.filter_stats['llm_rejected'] += 1
        
        return filtered_samples
    
    def save_filter_stats(self, retrain_id: str = None) -> None:
        """Save filter statistics to JSONL file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(FILTER_STATS_FILE), exist_ok=True)
            
            # Create stats entry
            stats_entry = {
                'timestamp': datetime.now().isoformat(),
                'retrain_id': retrain_id or f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                **self.filter_stats,
                'llm_stats': self.llm_validator.get_stats()
            }
            
            # Append to JSONL file
            with open(FILTER_STATS_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(stats_entry, ensure_ascii=False) + '\n')
            
            logger.info(f"Filter statistics saved to {FILTER_STATS_FILE}")
            
        except Exception as e:
            logger.error(f"Failed to save filter statistics: {e}")
    
    def get_filter_summary(self) -> str:
        """Get a human-readable summary of filtering results"""
        stats = self.filter_stats
        
        if stats['original_count'] == 0:
            return "No samples to filter"
        
        efficiency = stats['final_count'] / stats['original_count'] * 100
        
        summary = f"""
📊 Hybrid Filtering Summary
==========================
Original samples: {stats['original_count']}
Final samples: {stats['final_count']}
Efficiency: {efficiency:.1f}%

🔍 Filtering Breakdown:
  • Duplicates removed: {stats['duplicates_removed']}
  • Malformed removed: {stats['malformed_removed']}
  • Low confidence removed: {stats['low_confidence_removed']}
  • Borderline cases: {stats['borderline_count']}
    - LLM validated: {stats['llm_validated']}
    - LLM rejected: {stats['llm_rejected']}

🤖 LLM Validation:
  • Enabled: {self.llm_validator.enabled}
  • Model: {self.llm_validator.model}
  • Borderline range: {LLM_FALLBACK_RANGE[0]:.2f}-{LLM_FALLBACK_RANGE[1]:.2f}
"""
        
        return summary.strip()
    
    def get_detailed_stats(self) -> Dict:
        """Get detailed filtering statistics"""
        return {
            'filter_stats': self.filter_stats.copy(),
            'llm_stats': self.llm_validator.get_stats(),
            'config': {
                'min_confidence_threshold': MIN_CONFIDENCE_THRESHOLD,
                'llm_fallback_range': LLM_FALLBACK_RANGE,
                'min_sample_length': MIN_SAMPLE_LENGTH,
                'max_sample_length': MAX_SAMPLE_LENGTH
            }
        }
