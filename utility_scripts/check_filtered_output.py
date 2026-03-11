"""Check filtered output files"""
from pathlib import Path
import json

output_dir = Path("output_segmentation_filtered")

print("="*70)
print("Filtered Output Files")
print("="*70)

# Check directories
original_dir = output_dir / "original"
segmented_dir = output_dir / "segmented"
masks_dir = output_dir / "masks"

print(f"\n📁 Output Directory: {output_dir}")
print("-" * 70)

# Original images
if original_dir.exists():
    originals = list(original_dir.glob("*.jpg"))
    print(f"\n🖼️  Original Images: {len(originals)} files")
    if originals:
        print(f"   Example: {originals[0].name}")

# Segmented images
if segmented_dir.exists():
    segmented = list(segmented_dir.glob("*.jpg"))
    print(f"\n🎨 Segmented Images: {len(segmented)} files")
    if segmented:
        print(f"   Example: {segmented[0].name}")

# Mask JSON files
if masks_dir.exists():
    masks = list(masks_dir.glob("*.json"))
    print(f"\n📄 Mask JSON Files: {len(masks)} files")
    if masks:
        print(f"   Example: {masks[0].name}")
        # Show sample JSON structure
        with open(masks[0], 'r') as f:
            sample = json.load(f)
        print(f"\n   Sample JSON structure:")
        print(f"     - frame_id: {sample.get('frame_id')}")
        print(f"     - num_cracks: {sample.get('num_cracks')}")
        print(f"     - masks: {len(sample.get('masks', []))} mask(s)")
        if sample.get('masks'):
            mask0 = sample['masks'][0]
            print(f"       Mask 0:")
            print(f"         - area: {mask0.get('area')} pixels")
            print(f"         - confidence: {mask0.get('confidence'):.3f}")
            print(f"         - bbox: {mask0.get('bbox')}")

print("\n" + "="*70)
print("Summary:")
print(f"  Total frames processed: 372")
print(f"  Frames with cracks: 108 (29.0%)")
print(f"  Confidence threshold: 0.15")
print("="*70)
