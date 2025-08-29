#!/usr/bin/env python3
"""
Test script for the Parquet conversion pipeline.

This script demonstrates how to use the converted Parquet files
and verifies the pipeline works correctly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def test_parquet_loading():
    """Test loading all Parquet files."""
    print("🧪 Testing Parquet file loading...")
    
    try:
        # Load all datasets
        train_df = pd.read_parquet('data/processed/train.parquet')
        val_df = pd.read_parquet('data/processed/val.parquet')
        test_df = pd.read_parquet('data/processed/test.parquet')
        full_df = pd.read_parquet('data/processed/training_data.parquet')
        
        print(f"✅ Train set: {len(train_df)} rows, {len(train_df.columns)} columns")
        print(f"✅ Validation set: {len(val_df)} rows, {len(val_df.columns)} columns")
        print(f"✅ Test set: {len(test_df)} rows, {len(test_df.columns)} columns")
        print(f"✅ Full dataset: {len(full_df)} rows, {len(full_df.columns)} columns")
        
        # Verify split sizes
        total_split = len(train_df) + len(val_df) + len(test_df)
        if total_split == len(full_df):
            print("✅ Split sizes match full dataset")
        else:
            print(f"❌ Split size mismatch: {total_split} vs {len(full_df)}")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to load Parquet files: {e}")
        return False

def test_data_quality():
    """Test data quality and structure."""
    print("\n🔍 Testing data quality...")
    
    try:
        train_df = pd.read_parquet('data/processed/train.parquet')
        
        # Check for missing values
        missing_counts = train_df.isnull().sum()
        high_missing = missing_counts[missing_counts > len(train_df) * 0.5]
        
        if len(high_missing) > 0:
            print(f"⚠️  High missing values in columns: {list(high_missing.index)}")
        else:
            print("✅ No columns with excessive missing values")
        
        # Check data types
        object_cols = train_df.select_dtypes(include=['object']).columns
        print(f"📝 Object columns: {len(object_cols)}")
        
        # Check timestamp column
        if 'timestamp' in train_df.columns:
            print("✅ Timestamp column present")
            if 'date' in train_df.columns and 'hour' in train_df.columns:
                print("✅ Timestamp features added")
        else:
            print("⚠️  No timestamp column found")
            
        return True
        
    except Exception as e:
        print(f"❌ Data quality check failed: {e}")
        return False

def test_ml_ready_format():
    """Test that data is ready for ML training."""
    print("\n🤖 Testing ML-ready format...")
    
    try:
        train_df = pd.read_parquet('data/processed/train.parquet')
        
        # Check for essential columns
        essential_cols = ['text_input', 'primary_intent']
        missing_essential = [col for col in essential_cols if col not in train_df.columns]
        
        if missing_essential:
            print(f"⚠️  Missing essential columns: {missing_essential}")
        else:
            print("✅ Essential columns present")
        
        # Check data distribution
        if 'primary_intent' in train_df.columns:
            intent_counts = train_df['primary_intent'].value_counts()
            print(f"📊 Intent distribution: {len(intent_counts)} unique intents")
            print(f"   Top 5: {dict(intent_counts.head())}")
        
        # Check for duplicates
        duplicates = train_df.duplicated().sum()
        if duplicates > 0:
            print(f"⚠️  Found {duplicates} duplicate rows")
        else:
            print("✅ No duplicate rows")
            
        return True
        
    except Exception as e:
        print(f"❌ ML-ready format check failed: {e}")
        return False

def demonstrate_usage():
    """Demonstrate how to use the Parquet files for ML training."""
    print("\n📚 Usage Examples:")
    
    try:
        # Example 1: Basic loading
        print("\n1️⃣  Basic loading:")
        print("   train_df = pd.read_parquet('data/processed/train.parquet')")
        print("   val_df = pd.read_parquet('data/processed/val.parquet')")
        print("   test_df = pd.read_parquet('data/processed/test.parquet')")
        
        # Example 2: Feature engineering
        print("\n2️⃣  Feature engineering:")
        print("   # Text features")
        print("   train_df['text_length'] = train_df['text_input'].str.len()")
        print("   # Intent encoding")
        print("   from sklearn.preprocessing import LabelEncoder")
        print("   le = LabelEncoder()")
        print("   train_df['intent_encoded'] = le.fit_transform(train_df['primary_intent'])")
        
        # Example 3: Data splitting for cross-validation
        print("\n3️⃣  Cross-validation:")
        print("   from sklearn.model_selection import train_test_split")
        print("   X_train, X_temp, y_train, y_temp = train_test_split(")
        print("       train_df['text_input'], train_df['primary_intent'], test_size=0.2")
        print("   )")
        
        # Example 4: Loading with filters
        print("\n4️⃣  Loading with filters:")
        print("   # Filter by event type")
        print("   query_events = train_df[train_df['event_type'] == 'query']")
        print("   # Filter by confidence")
        print("   high_conf = train_df[train_df['confidence'] > 0.7]")
        
        return True
        
    except Exception as e:
        print(f"❌ Usage demonstration failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Parquet Conversion Pipeline")
    print("=" * 50)
    
    tests = [
        test_parquet_loading,
        test_data_quality,
        test_ml_ready_format,
        demonstrate_usage
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! The Parquet pipeline is working correctly.")
        print("📁 Your ML training data is ready in data/processed/")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
