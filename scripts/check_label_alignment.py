"""
Check alignment between dataset labels and taxonomy labels.
This helps ensure your model predictions and taxonomy categories are speaking the same language.

Uses the new reusable label functions from utils.label_utils for consistency across the project.
"""

import sys
from pathlib import Path

# Add project root to PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from utils.label_utils import (
    load_taxonomy,
    load_dataset_labels,
    get_label_mappings_from_taxonomy_and_dataset,
    get_reverse_label_mappings,
    check_label_consistency
)


def main():
    """Main function to check label alignment."""
    print("🔍 Checking Label Alignment Between Dataset and Taxonomy")
    print("=" * 60)
    
    # Load taxonomy and dataset labels using reusable functions
    print("📂 Loading taxonomy...")
    taxonomy = load_taxonomy()
    print(f"   ✅ Loaded {len(taxonomy)} taxonomy categories")
    
    print("📂 Loading dataset labels...")
    dataset_labels = load_dataset_labels()
    print(f"   ✅ Loaded {len(dataset_labels)} unique dataset labels")
    
    # Build unified mappings using the new function
    print("🔗 Building unified label mappings...")
    LABEL_TO_ID = get_label_mappings_from_taxonomy_and_dataset(taxonomy, dataset_labels)
    ID_TO_LABEL = get_reverse_label_mappings(LABEL_TO_ID)
    print(f"   ✅ Created unified mapping with {len(LABEL_TO_ID)} labels")
    
    # Use the consistency check function from label_utils
    print("\n📊 Running consistency check...")
    consistency_report = check_label_consistency()
    
    # Display results
    print("\n" + "=" * 60)
    print("📋 LABEL ALIGNMENT REPORT")
    print("=" * 60)
    
    print(f"📂 Taxonomy categories: {len(taxonomy)}")
    print(f"📂 Dataset labels: {len(dataset_labels)}")
    print(f"🔗 Unified labels: {len(LABEL_TO_ID)}")
    
    print(f"\n📊 Consistency Summary:")
    print(f"   • Common labels: {len(consistency_report['common'])}")
    print(f"   • Taxonomy only: {len(consistency_report['taxonomy_only'])}")
    print(f"   • Dataset only: {len(consistency_report['dataset_only'])}")
    
    # Show example mappings
    print(f"\n🔗 Example Label Mappings (first 10):")
    for i, (label, label_id) in enumerate(list(LABEL_TO_ID.items())[:10]):
        print(f"   {i+1:2d}. {label:30s} → ID {label_id}")
    
    # Show some dataset labels
    print(f"\n📝 Sample Dataset Labels (first 15):")
    for i, label in enumerate(sorted(dataset_labels)[:15]):
        print(f"   {i+1:2d}. {label}")
    
    # Show taxonomy categories
    print(f"\n🏷️  Taxonomy Categories:")
    for i, category in enumerate(sorted(taxonomy.keys())):
        print(f"   {i+1:2d}. {category}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if len(consistency_report['common']) == 0:
        print("   ❌ CRITICAL: No common labels between taxonomy and dataset!")
        print("   🔧 Action: Review taxonomy structure and dataset labeling")
        print("   🔧 Action: Consider updating taxonomy to match dataset labels")
    elif len(consistency_report['common']) < 5:
        print("   ⚠️  WARNING: Very few common labels")
        print("   🔧 Action: Investigate label naming conventions")
    else:
        print("   ✅ Good label alignment")
    
    print(f"\n🎯 Next Steps:")
    print("   • Use load_dataset_labels() for consistent dataset label loading")
    print("   • Use get_label_mappings_from_taxonomy_and_dataset() for unified mappings")
    print("   • Consider running this script after taxonomy updates")


if __name__ == "__main__":
    main()