"""
Crack Segmentation Dataset Preparation - Fast Multi-threaded Version
Downloads images from cracknet-images.ndjson with concurrent downloads
"""

import json
import os
import urllib.request
import urllib.parse
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

# Global counters for progress tracking
downloaded_count = 0
failed_count = 0
total_count = 0
lock = threading.Lock()


def download_image(args):
    """Download single image"""
    global downloaded_count, failed_count, total_count
    
    url, save_path, filename = args
    
    # Skip if already exists
    if os.path.exists(save_path):
        with lock:
            downloaded_count += 1
        return True, filename
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        req = urllib.request.Request(url, headers=headers)
        
        # Set timeout and retry
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    with open(save_path, 'wb') as f:
                        f.write(response.read())
                
                with lock:
                    downloaded_count += 1
                    
                    # Print progress every 50 images
                    if downloaded_count % 50 == 0:
                        progress = (downloaded_count / total_count) * 100
                        print(f"Progress: {downloaded_count}/{total_count} ({progress:.1f}%)")
                
                return True, filename
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise
                time.sleep(0.5)  # Wait before retry
                
    except Exception as e:
        with lock:
            failed_count += 1
        return False, filename


def convert_segments_to_yolo(segments, width, height):
    """
    Convert segments format to YOLO format
    """
    yolo_annotations = []
    
    for seg in segments:
        if len(seg) < 3:
            continue
            
        class_id = int(seg[0])
        points = seg[1:]
        
        # Normalize points
        normalized_points = []
        for i in range(0, len(points), 2):
            if i + 1 < len(points):
                x = points[i] / width
                y = points[i + 1] / height
                normalized_points.extend([x, y])
        
        if normalized_points:
            yolo_annotations.append([class_id] + normalized_points)
    
    return yolo_annotations


def prepare_dataset(ndjson_path, output_dir, max_workers=20):
    """
    Prepare dataset from ndjson file with multi-threading
    """
    global total_count, downloaded_count, failed_count
    
    output_dir = Path(output_dir)
    
    # Create directories
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    
    (images_dir / 'train').mkdir(parents=True, exist_ok=True)
    (images_dir / 'val').mkdir(parents=True, exist_ok=True)
    (labels_dir / 'train').mkdir(parents=True, exist_ok=True)
    (labels_dir / 'val').mkdir(parents=True, exist_ok=True)
    
    # Read ndjson file
    print("Reading ndjson file...")
    with open(ndjson_path, 'r') as f:
        lines = f.readlines()
    
    # Parse dataset info
    dataset_info = json.loads(lines[0])
    print(f"Dataset: {dataset_info.get('name', 'Unknown')}")
    print(f"Task: {dataset_info.get('task', 'Unknown')}")
    print(f"Total entries: {len(lines) - 1}")
    
    # Collect all download tasks
    download_tasks = []
    image_data_list = []
    
    print("Preparing download list...")
    for line in lines[1:]:
        try:
            data = json.loads(line)
            
            if data.get('type') != 'image':
                continue
            
            filename = data['file']
            url = data['url']
            width = data['width']
            height = data['height']
            split = data.get('split', 'train')
            annotations = data.get('annotations', {})
            
            # Determine output directories
            if split == 'val':
                img_dir = images_dir / 'val'
                lbl_dir = labels_dir / 'val'
            else:
                img_dir = images_dir / 'train'
                lbl_dir = labels_dir / 'train'
            
            img_path = img_dir / filename
            
            # Add to download tasks
            download_tasks.append((url, str(img_path), filename))
            
            # Store data for label generation
            image_data_list.append({
                'filename': filename,
                'width': width,
                'height': height,
                'annotations': annotations,
                'label_dir': lbl_dir
            })
            
        except Exception as e:
            print(f"Error parsing entry: {e}")
            continue
    
    total_count = len(download_tasks)
    print(f"\nTotal images to download: {total_count}")
    print(f"Using {max_workers} concurrent threads\n")
    
    # Download images with ThreadPoolExecutor
    print("Starting download...")
    start_time = time.time()
    
    failed_downloads = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(download_image, task): task[2] for task in download_tasks}
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            success, filename = future.result()
            if not success:
                failed_downloads.append(filename)
    
    download_time = time.time() - start_time
    print(f"\nDownload complete in {download_time:.1f} seconds")
    print(f"Successfully downloaded: {downloaded_count}")
    print(f"Failed downloads: {failed_count}")
    
    # Generate labels for successfully downloaded images
    print("\nGenerating labels...")
    label_count = 0
    
    for img_data in image_data_list:
        try:
            filename = img_data['filename']
            width = img_data['width']
            height = img_data['height']
            annotations = img_data['annotations']
            lbl_dir = img_data['label_dir']
            
            segments = annotations.get('segments', [])
            if segments:
                yolo_annotations = convert_segments_to_yolo(segments, width, height)
                
                # Save YOLO format annotation
                label_filename = filename.replace('.jpg', '.txt').replace('.png', '.txt')
                label_path = lbl_dir / label_filename
                
                with open(label_path, 'w') as f:
                    for ann in yolo_annotations:
                        line = ' '.join([str(x) for x in ann])
                        f.write(line + '\n')
                
                label_count += 1
        except Exception as e:
            print(f"Error generating label for {filename}: {e}")
    
    print(f"Generated {label_count} label files")
    
    # Count train/val split
    train_images = len(list((images_dir / 'train').glob('*.jpg')))
    val_images = len(list((images_dir / 'val').glob('*.jpg')))
    
    print(f"\nDataset preparation complete!")
    print(f"Train images: {train_images}")
    print(f"Val images: {val_images}")
    
    if failed_downloads:
        print(f"\nFailed downloads ({len(failed_downloads)}):")
        for f in failed_downloads[:10]:  # Show first 10
            print(f"  - {f}")
    
    return train_images, val_images


def create_dataset_yaml(output_dir):
    """Create dataset.yaml for YOLO training"""
    yaml_content = f"""# Crack Segmentation Dataset
path: {output_dir}  # dataset root dir
train: images/train  # train images
val: images/val  # val images

# Classes
names:
  0: crack

# Number of classes
nc: 1
"""
    
    yaml_path = Path(output_dir) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nDataset YAML created: {yaml_path}")


if __name__ == "__main__":
    # Configuration
    NDJSON_PATH = "trainData/cracknet-images.ndjson"
    OUTPUT_DIR = "trainData/crack_segmentation"
    MAX_WORKERS = 20  # Number of concurrent downloads
    
    print("="*60)
    print("Fast Crack Segmentation Dataset Preparation")
    print("="*60)
    
    # Prepare dataset
    train_count, val_count = prepare_dataset(NDJSON_PATH, OUTPUT_DIR, MAX_WORKERS)
    
    # Create YAML file
    create_dataset_yaml(OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("Next steps:")
    print("1. Verify downloaded images in trainData/crack_segmentation/images/")
    print("2. Verify labels in trainData/crack_segmentation/labels/")
    print("3. Train YOLO model with:")
    print("   python train_crack_seg.py")
    print("="*60)
