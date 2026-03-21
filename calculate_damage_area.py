# -*- coding: utf-8 -*-
"""
计算损伤区域表面积脚本

读取 damage_only.ply 点云，重建表面并计算总面积
"""

import open3d as o3d
import numpy as np
import os
from pathlib import Path


def calculate_damage_surface_area(
    ply_path: str,
    voxel_size: float = 0.01,
    ball_pivoting_search_radius: float = 0.05
) -> dict:
    """
    计算损伤点云的表面积

    Args:
        ply_path: damage_only.ply 文件路径
        voxel_size: 体素大小（降采样用）
        ball_pivoting_search_radius: 球旋转算法搜索半径

    Returns:
        包含计算结果的字典
    """
    print("=" * 60)
    print("Damage Surface Area Calculator")
    print("=" * 60)

    if not os.path.exists(ply_path):
        print(f"[Error] File not found: {ply_path}")
        return None

    print(f"\n[Info] Reading point cloud: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    original_points = len(pcd.points)
    print(f"[Info] Original points: {original_points}")

    if original_points == 0:
        print("[Error] Point cloud is empty!")
        return None

    original_colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    print(f"\n[Processing] Downsampling with voxel size: {voxel_size}...")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    downsampled_points = len(pcd_down.points)
    print(f"[Info] Downsampled points: {downsampled_points}")

    print(f"\n[Processing] Estimating normals...")
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 4,
            max_nn=30
        )
    )
    pcd_down.orient_normals_consistent_tangent_plane(k=15)

    print(f"\n[Processing] Surface reconstruction (Ball Pivoting)...")
    radii = [
        ball_pivoting_search_radius * 0.5,
        ball_pivoting_search_radius,
        ball_pivoting_search_radius * 2
    ]

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd_down,
        o3d.utility.DoubleVector(radii)
    )

    triangle_count = len(mesh.triangles)
    print(f"[Info] Generated triangles: {triangle_count}")

    if triangle_count == 0:
        print("[Warning] Ball pivoting failed, trying Poisson reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_down,
            depth=8,
            width=0,
            scale=1.1,
            linear_fit=False
        )

        triangle_count = len(mesh.triangles)
        print(f"[Info] Poisson generated {triangle_count} triangles")

        vertices_to_remove = densities < np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print(f"[Info] After removing low-density vertices: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

    mesh.compute_vertex_normals()

    total_surface_area = mesh.get_surface_area()
    surface_area_cm2 = total_surface_area * 10000

    bounding_box = mesh.get_axis_aligned_bounding_box()
    bbox_volume = bounding_box.volume()
    bbox_extent = bounding_box.get_extent()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Total Surface Area: {total_surface_area:.6f} m²")
    print(f"  Total Surface Area: {surface_area_cm2:.2f} cm²")
    print(f"  Number of Triangles: {triangle_count}")
    print(f"  Bounding Box: {bbox_extent[0]:.3f} x {bbox_extent[1]:.3f} x {bbox_extent[2]:.3f} m")
    print(f"  Bounding Volume: {bbox_volume:.6f} m³")
    print("=" * 60)

    result = {
        "surface_area_m2": total_surface_area,
        "surface_area_cm2": surface_area_cm2,
        "triangle_count": triangle_count,
        "original_points": original_points,
        "downsampled_points": downsampled_points,
        "bounding_box_extent": bbox_extent.tolist(),
        "bounding_box_volume": bbox_volume,
        "mesh": mesh
    }

    return result


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate damage surface area from point cloud",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python calculate_damage_area.py
  python calculate_damage_area.py --ply "path/to/damage_only.ply"
  python calculate_damage_area.py --ply "path/to/damage_only.ply" --voxel-size 0.005
        """
    )

    default_ply = r"data/damage_detection/HD2K_SN36245620_15-17-12/HD2K_SN36245620_15-17-12_damage_only.ply"

    parser.add_argument(
        '--ply',
        type=str,
        default=default_ply,
        help=f'Path to damage_only.ply file (default: {default_ply})'
    )

    parser.add_argument(
        '--voxel-size',
        type=float,
        default=0.01,
        help='Voxel size for downsampling (default: 0.01 m)'
    )

    parser.add_argument(
        '--radius',
        type=float,
        default=0.05,
        help='Ball pivoting search radius (default: 0.05 m)'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize the reconstructed surface'
    )

    args = parser.parse_args()

    result = calculate_damage_surface_area(
        ply_path=args.ply,
        voxel_size=args.voxel_size,
        ball_pivoting_search_radius=args.radius
    )

    if result is not None and args.visualize:
        print("\n[Info] Visualizing reconstructed surface...")
        mesh = result["mesh"]
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries(
            [mesh],
            window_name="Damage Surface Reconstruction",
            width=1280,
            height=720
        )


if __name__ == "__main__":
    main()
