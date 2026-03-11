"""Check output files"""
from pathlib import Path

output_dir = Path("output_segmentation")

print("="*70)
print("Output Files")
print("="*70)

# Check video
video_path = output_dir / "segmentation_result.mp4"
if video_path.exists():
    size_mb = video_path.stat().st_size / (1024 * 1024)
    print(f"\n📹 Video: {video_path}")
    print(f"   Size: {size_mb:.2f} MB")

# Check frames
frames_dir = output_dir / "frames"
if frames_dir.exists():
    frames = list(frames_dir.glob("*.jpg"))
    print(f"\n🖼️  Frames: {frames_dir}")
    print(f"   Count: {len(frames)} frames")
    if frames:
        # Show first and last frame
        print(f"   First: {frames[0].name}")
        print(f"   Last: {frames[-1].name}")

print("\n" + "="*70)
