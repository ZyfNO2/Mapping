# ZED SDK - Spatial Mapping

This sample shows how to map your environment and process point cloud data, with additional support for YOLO-based image segmentation.

## Getting Started

### Prerequisites
- Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/) and [pyZED Package](https://www.stereolabs.com/docs/app-development/python/install/)
- Install [Open3D](http://www.open3d.org/) for point cloud processing and visualization
- Check the [Documentation](https://www.stereolabs.com/docs/)

## Deployment Steps

### 1. Clone the Repository
```bash
git clone <repository-url>
cd python
```

### 2. Create Python Environment
```bash
# Using conda (recommended)
conda create -n zed python=3.10
conda activate zed

# Or using venv
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install ZED SDK Python API
Follow the [official guide](https://www.stereolabs.com/docs/app-development/python/install/) to install pyzed.

### 5. Download YOLO Model (Optional)
For segmentation features, download the YOLO model:
```bash
python -c "from ultralytics import YOLO; YOLO('yolo26n-seg.pt')"
```

## Run the program

### Spatial Mapping (Real-time)
To run the program with real-time camera input:
```bash
python spatial_mapping.py
```

If you wish to run the program from an input_svo_file, or an IP address, or specify a resolution:
```bash
python spatial_mapping.py --input_svo_file <input_svo_file> --ip_address <ip_address> --resolution <resolution>
```
Arguments:
  - `--input_svo_file` A path to an existing .svo file, that will be playbacked. If this parameter and ip_address are not specified, the soft will use the camera wired as default.
  - `--ip_address` IP Address, in format a.b.c.d:port or a.b.c.d. If specified, the soft will try to connect to the IP.
  - `--resolution` Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA

### Spatial Mapping (Offline)
To process an SVO file and generate high-quality point cloud:
```bash
python spatial_mapping_offline.py
```
This will:
- Process the SVO file and generate a high-quality point cloud
- Save the point cloud as PLY format: `data/point_cloud_gen_high_quality.ply`
- Save the point cloud as CSV format: `data/point_cloud_gen_high_quality.csv` (includes xyz, normal, and rgb data)

### First Frame Center Point Extraction
To extract the first frame from an SVO file and find the 3D coordinates of the center point:
```bash
python first_frame_center_point.py
```
This will:
- Extract the first frame from the SVO file
- Calculate the 3D coordinates of the center point
- Save the annotated first frame image
- Visualize the point cloud with a magenta sphere at the center coordinate

### Point Cloud Processing
To process the generated point cloud with filtering and clustering:
```bash
python point_cloud_processing.py data/point_cloud_gen_high_quality.ply data/point_cloud_processed_no_downsample.ply
```
Optional arguments:
  - `--density_radius` Search radius for density filtering (default: 0.06)
  - `--min_density` Minimum local density for density filtering (default: 10)
  - `--filter_type` Filter type: statistical or radius (default: statistical)
  - `--nb_neighbors` Number of neighbors for statistical filter (default: 20)
  - `--std_ratio` Standard deviation ratio for statistical filter (default: 2.0)
  - `--visualize` Visualize the point cloud at each step

This will:
- Apply density filtering to remove discrete points
- Perform statistical or radius outlier removal
- Estimate normals for each point
- Remove outliers based on curvature
- Cluster and remove small fragments
- Save the processed point cloud as PLY format
- Save the processed point cloud as CSV format (includes xyz, normal, and rgb data)

### Point Cloud Distance Analysis
To analyze distances in the point cloud:
```bash
python point_cloud_distance_analysis.py
```

### PLY Viewer
To visualize point cloud files with grid analysis and selection:
```bash
python ply_viewer.py
```
Features:
- 3x3 grid layout analysis showing point distribution
- Displays point count and first point coordinates for each grid cell
- Interactive point selection with mouse
- Supports both PLY and CSV formats

### Point Cloud Converter
To convert between different point cloud formats:
```bash
python point_cloud_converter.py <input_file> <output_file>
```

### YOLO Segmentation Test
To test YOLO26n-seg segmentation model:
```bash
python -c "from ultralytics import YOLO; model = YOLO('yolo26n-seg.pt'); results = model('image.jpg'); results[0].save('result.jpg')"
```

## Features
- **Real-time Mapping**: Press 'Spacebar' to start/stop the mapping process
- **Real-time Overlay**: Mesh overlay on the image
- **Textures and Post-filters**: Can be applied to the mesh
- **CSV Export**: Point clouds are exported with xyz coordinates, normal vectors, and RGB colors
- **Grid Analysis**: 3x3 grid layout for analyzing point distribution
- **Interactive Selection**: Box selection functionality in PLY viewer
- **First Frame Analysis**: Extract and analyze the first frame from SVO files
- **YOLO Segmentation**: Support for instance segmentation using YOLO models
- **Final Mesh/Point Cloud**: Automatically saved after processing

## Output Files
- `data/point_cloud_gen_high_quality.ply` - Raw high-quality point cloud
- `data/point_cloud_gen_high_quality.csv` - Raw point cloud in CSV format (x,y,z,nx,ny,nz,r,g,b)
- `data/point_cloud_processed_no_downsample.ply` - Processed point cloud
- `data/point_cloud_processed_no_downsample.csv` - Processed point cloud in CSV format (x,y,z,nx,ny,nz,r,g,b)
- `data/first_frame.jpg` - First frame extracted from SVO file
- `data/first_frame_annotated.jpg` - Annotated first frame with center point
- `data/seg_masks.json` - YOLO segmentation mask data

## Project Structure
```
python/
├── spatial_mapping.py              # Real-time spatial mapping
├── spatial_mapping_offline.py      # Offline SVO processing
├── first_frame_center_point.py     # First frame extraction and center point analysis
├── point_cloud_processing.py       # Point cloud filtering and clustering
├── point_cloud_distance_analysis.py # Point cloud distance analysis
├── point_cloud_converter.py        # Format conversion utility
├── ply_viewer.py                   # Point cloud visualization
├── requirements.txt                # Python dependencies
├── yolo26n-seg.pt                  # YOLO segmentation model (auto-downloaded)
├── ultralytics/                    # YOLO source code (cloned)
├── crack-detection/                # Crack detection dataset
└── data/                           # Output directory
```

## Support
If you need assistance go to our Community site at https://community.stereolabs.com/
