#!/usr/bin/env python3
"""
JSONL to Parquet Conversion Script

This script converts all JSONL log files under data/logs/ into a structured
Parquet dataset with automatic train/val/test splits for machine learning.

Features:
- Loads all JSONL files from data/logs/
- Normalizes nested JSON structures
- Sorts by timestamp (if available)
- Creates 80/10/10 train/val/test splits
- Saves as Parquet files for efficient ML training
- Uses PyArrow engine for optimal performance
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import random
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load events from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of parsed JSON events
    """
    events = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    event = json.loads(line)
                    events.append(event)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
                    continue
                    
        logger.info(f"Loaded {len(events)} events from {file_path.name}")
        return events
        
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return []

def normalize_events(events: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize events into a flat DataFrame structure.
    
    Args:
        events: List of event dictionaries
        
    Returns:
        Normalized DataFrame
    """
    if not events:
        return pd.DataFrame()
    
    # Use pandas json_normalize to handle nested structures
    try:
        df = pd.json_normalize(events, sep='_')
        logger.info(f"Normalized {len(df)} events into DataFrame with {len(df.columns)} columns")
        
        # Clean and validate data types for Parquet compatibility
        df = clean_dataframe_for_parquet(df)
        
        return df
    except Exception as e:
        logger.error(f"Failed to normalize events: {e}")
        return pd.DataFrame()

def clean_dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame to ensure Parquet compatibility.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame compatible with Parquet
    """
    try:
        # Convert problematic columns to strings to avoid type issues
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains mixed types
                sample_values = df[col].dropna().head(100)
                if len(sample_values) > 0:
                    # Convert to string representation to preserve data
                    df[col] = df[col].astype(str)
        
        # Handle specific problematic columns
        if 'all_intents' in df.columns:
            # Convert all_intents to string representation
            df['all_intents'] = df['all_intents'].astype(str)
        
        if 'recommendations' in df.columns:
            # Convert recommendations to string representation
            df['recommendations'] = df['recommendations'].astype(str)
        
        logger.info("Cleaned DataFrame for Parquet compatibility")
        return df
        
    except Exception as e:
        logger.error(f"Failed to clean DataFrame: {e}")
        return df

def add_timestamp_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add timestamp-based features for better sorting and analysis.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with timestamp features
    """
    if 'timestamp' not in df.columns:
        logger.warning("No timestamp column found, skipping timestamp features")
        return df
    
    try:
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Add useful timestamp features
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Sort by timestamp (oldest first)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info("Added timestamp features and sorted by timestamp")
        return df
        
    except Exception as e:
        logger.error(f"Failed to add timestamp features: {e}")
        return df

def create_train_val_test_splits(df: pd.DataFrame, 
                                train_ratio: float = 0.8,
                                val_ratio: float = 0.1,
                                test_ratio: float = 0.1,
                                random_state: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Create train/validation/test splits from the dataset.
    
    Args:
        df: Input DataFrame
        train_ratio: Proportion for training (default: 0.8)
        val_ratio: Proportion for validation (default: 0.1)
        test_ratio: Proportion for testing (default: 0.1)
        random_state: Random seed for reproducible splits
        
    Returns:
        Dictionary with 'train', 'val', 'test' DataFrames
    """
    if len(df) == 0:
        logger.warning("Empty DataFrame, returning empty splits")
        return {'train': pd.DataFrame(), 'val': pd.DataFrame(), 'test': pd.DataFrame()}
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Set random seed for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Shuffle the data
    df_shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    # Calculate split indices
    n_total = len(df_shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Create splits
    train_df = df_shuffled.iloc[:n_train].copy()
    val_df = df_shuffled.iloc[n_train:n_train + n_val].copy()
    test_df = df_shuffled.iloc[n_train + n_val:].copy()
    
    # Log split information
    logger.info(f"Dataset split: train={len(train_df)} ({len(train_df)/n_total:.1%}), "
                f"val={len(val_df)} ({len(val_df)/n_total:.1%}), "
                f"test={len(test_df)} ({len(test_df)/n_total:.1%})")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

def save_parquet_files(df: pd.DataFrame, 
                      splits: Dict[str, pd.DataFrame], 
                      output_dir: Path) -> None:
    """
    Save the full dataset and splits as Parquet files.
    
    Args:
        df: Full dataset DataFrame
        splits: Dictionary with train/val/test DataFrames
        output_dir: Directory to save Parquet files
    """
    try:
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full dataset
        full_dataset_path = output_dir / "training_data.parquet"
        df.to_parquet(full_dataset_path, engine='pyarrow', index=False)
        logger.info(f"✅ Saved full dataset: {full_dataset_path}")
        
        # Save splits
        for split_name, split_df in splits.items():
            if len(split_df) > 0:
                split_path = output_dir / f"{split_name}.parquet"
                split_df.to_parquet(split_path, engine='pyarrow', index=False)
                logger.info(f"✅ Saved {split_name} split: {split_path}")
            else:
                logger.warning(f"⚠️ {split_name} split is empty, skipping")
                
    except Exception as e:
        logger.error(f"Failed to save Parquet files: {e}")
        raise

def print_dataset_summary(df: pd.DataFrame, splits: Dict[str, pd.DataFrame]) -> None:
    """
    Print a comprehensive summary of the dataset.
    
    Args:
        df: Full dataset DataFrame
        splits: Dictionary with train/val/test DataFrames
    """
    print("\n" + "="*60)
    print("📊 DATASET CONVERSION SUMMARY")
    print("="*60)
    
    # Basic statistics
    print(f"📈 Total Events: {len(df):,}")
    print(f"🔢 Columns: {len(df.columns)}")
    
    # Event type breakdown (if available)
    if 'event_type' in df.columns:
        event_counts = df['event_type'].value_counts()
        print(f"\n📋 Event Types:")
        for event_type, count in event_counts.items():
            print(f"   • {event_type}: {count:,}")
    
    # Intent breakdown (if available)
    if 'primary_intent' in df.columns:
        intent_counts = df['primary_intent'].value_counts().head(10)
        print(f"\n🎯 Top Intents:")
        for intent, count in intent_counts.items():
            print(f"   • {intent}: {count:,}")
    
    # Split summary
    print(f"\n✂️  Dataset Splits:")
    for split_name, split_df in splits.items():
        print(f"   • {split_name.capitalize()}: {len(split_df):,} events")
    
    # File paths
    print(f"\n💾 Output Files:")
    print(f"   • Full dataset: data/processed/training_data.parquet")
    print(f"   • Train split: data/processed/train.parquet")
    print(f"   • Validation split: data/processed/val.parquet")
    print(f"   • Test split: data/processed/test.parquet")
    
    print("\n" + "="*60)

def convert_logs(logs_dir: Path = Path("data/logs"),
                output_dir: Path = Path("data/processed"),
                train_ratio: float = 0.8,
                val_ratio: float = 0.1,
                test_ratio: float = 0.1) -> bool:
    """
    Main conversion function that processes all JSONL logs.
    
    Args:
        logs_dir: Directory containing JSONL log files
        output_dir: Directory to save Parquet files
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        logger.info("🚀 Starting JSONL to Parquet conversion...")
        
        # Find all JSONL files
        jsonl_files = list(logs_dir.glob("*.jsonl"))
        if not jsonl_files:
            logger.error(f"No JSONL files found in {logs_dir}")
            return False
        
        logger.info(f"📁 Found {len(jsonl_files)} JSONL files: {[f.name for f in jsonl_files]}")
        
        # Load all events
        all_events = []
        for file_path in jsonl_files:
            events = load_jsonl(file_path)
            all_events.extend(events)
        
        if not all_events:
            logger.error("No events loaded from any JSONL files")
            return False
        
        logger.info(f"📊 Loaded {len(all_events)} total events")
        
        # Normalize events
        df = normalize_events(all_events)
        if df.empty:
            logger.error("Failed to normalize events")
            return False
        
        # Add timestamp features
        df = add_timestamp_features(df)
        
        # Create splits
        splits = create_train_val_test_splits(
            df, train_ratio, val_ratio, test_ratio
        )
        
        # Save Parquet files
        save_parquet_files(df, splits, output_dir)
        
        # Print summary
        print_dataset_summary(df, splits)
        
        logger.info("🎉 Conversion completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        return False

if __name__ == "__main__":
    # Run the conversion
    success = convert_logs()
    
    if success:
        print("\n✅ Conversion completed successfully!")
        print("📚 You can now use the Parquet files for ML training:")
        print("   • pd.read_parquet('data/processed/train.parquet')")
        print("   • pd.read_parquet('data/processed/val.parquet')")
        print("   • pd.read_parquet('data/processed/test.parquet')")
    else:
        print("\n❌ Conversion failed. Check the logs above for details.")
        exit(1)
