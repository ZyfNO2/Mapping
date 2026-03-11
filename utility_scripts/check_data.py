"""Check data folder"""
from pathlib import Path

data_dir = Path("data/HD2K_SN36245620_15-17-12")
print(f"Checking: {data_dir}")
print(f"Exists: {data_dir.exists()}")

if data_dir.exists():
    print("\nFiles:")
    for f in data_dir.iterdir():
        if f.is_file():
            size = f.stat().st_size
            print(f"  {f.name}: {size} bytes")
