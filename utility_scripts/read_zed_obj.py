"""Read ZED OBJ file and convert to PLY"""
import numpy as np
from pathlib import Path

obj_path = Path("data/HD2K_SN36245620_15-17-12/point_cloud_gen_high_quality.obj")

print(f"Reading: {obj_path}")

vertices = []
colors = []

with open(obj_path, 'r') as f:
    for line in f:
        if line.startswith('v '):
            parts = line.strip().split()
            # v x y z r g b
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            # ZED OBJ格式: v x y z r g b (颜色是0-1浮点数)
            r, g, b = float(parts[4]), float(parts[5]), float(parts[6])
            vertices.append([x, y, z])
            colors.append([r, g, b])

vertices = np.array(vertices)
colors = np.array(colors)

print(f"Loaded {len(vertices)} vertices")

# Save as PLY using Open3D
import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vertices)
pcd.colors = o3d.utility.Vector3dVector(colors)

output_path = "data/HD2K_SN36245620_15-17-12/point_cloud_gen_high_quality.ply"
o3d.io.write_point_cloud(output_path, pcd)
print(f"Saved to: {output_path}")
print(f"File size: {Path(output_path).stat().st_size / (1024*1024):.2f} MB")
