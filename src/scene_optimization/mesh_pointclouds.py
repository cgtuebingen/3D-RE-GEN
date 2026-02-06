#!/usr/bin/env python3
"""
Standalone script to create meshes from saved pointclouds using Poisson surface reconstruction.
This script runs separately from the main pipeline to avoid Open3D/PyColmap conflicts.

Usage:
    python scripts/mesh_pointclouds.py --input output/pointclouds --output output/pointclouds/meshed
"""

import argparse
import os
import glob
import numpy as np
import logging
import torch
import trimesh

logging.basicConfig(level=logging.INFO, format="%(message)s")

# path append src
import sys

sys.path.append("src/")
from utils.global_utils import B2P, save_point_cloud


def set_vggt_cloud(vggt_cloud_path: str, vggt_scene_scale: float = 5.0, device="cpu"):
    """
    Load and transform VGGT point cloud with a single transformation matrix.

    Applies one transformation matrix to raw VGGT points:
    [ scale    0        0      0]
    [ 0       -scale    0      0]
    [ 0        0       -scale  0]
    [ 0        0        0      1]

    This directly transforms from VGGT coordinate system to the target coordinate system.

    Args:
        vggt_cloud_path: Path to the VGGT point cloud PLY file
        vggt_scene_scale: Scale factor (default: 5.0)
        device: torch device to use

    Returns:
        Transformed point cloud tensor (N, 3)
    """
    # Load raw VGGT cloud (PLY)
    pc_trimesh = trimesh.load(vggt_cloud_path)
    points = torch.from_numpy(pc_trimesh.vertices).float().to(device)  # (N, 3)

    logging.debug(f"[DEBUG] Raw VGGT points loaded:")
    logging.debug(f"  Shape: {points.shape}")
    logging.debug(f"  Bounds: min={points.min(dim=0).values}, max={points.max(dim=0).values}")

    # Apply single transformation matrix:
    # [ scale    0        0     ]
    # [ 0       -scale    0     ]
    # [ 0        0       -scale ]
    transform_matrix = torch.tensor(
        [
            [vggt_scene_scale, 0.0, 0.0],
            [0.0, -vggt_scene_scale, 0.0],
            [0.0, 0.0, -vggt_scene_scale],
        ],
        dtype=torch.float32,
        device=device,
    )

    points_transformed = points @ transform_matrix.T

    logging.debug(f"After single transformation (scale={vggt_scene_scale}):")
    logging.debug(
        f"  Bounds: min={points_transformed.min(dim=0).values}, max={points_transformed.max(dim=0).values}"
    )

    logging.info(
        f"Transformed point cloud: {points_transformed.shape} "
        f"with min/max bounds {points_transformed.min(dim=0).values} / {points_transformed.max(dim=0).values}"
    )

    return points_transformed


# remesh with trimesh
def clean_and_remesh(mesh, make_quads: bool = False, remesh_percentage: float = 0.5):
    """
    Clean and remesh an Open3D mesh using trimesh for repair operations.

    Performs the following operations:
    1. Convert Open3D mesh to trimesh
    2. Fill holes in the mesh
    3. Fix winding and normals
    4. Check and repair broken faces
    5. Remesh to reduce face count
    6. Convert faces to quads where possible
    7. Return trimesh object for direct saving

    Args:
        mesh: Open3D TriangleMesh object to clean and remesh

    Returns:
        trimesh.Trimesh: Cleaned and remeshed trimesh object
    """
    import open3d as o3d

    logging.info("  Cleaning and remeshing mesh...")

    # Convert Open3D mesh to trimesh
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    # Check if mesh has vertex colors
    has_colors = mesh.has_vertex_colors()
    vertex_colors = np.asarray(mesh.vertex_colors) if has_colors else None

    # Create trimesh object
    try:
        tmesh = trimesh.Trimesh(
            vertices=vertices, faces=faces, process=False, validate=False
        )
    except Exception as e:
        logging.error(f"    - Failed to create trimesh object: {e}")
        return None

    # 1. Fill holes
    logging.info("    - Filling holes...")
    try:
        trimesh.repair.fill_holes(tmesh)
    except Exception as e:
        logging.warning(f"    - Fill holes failed: {e}, continuing...")

    # 2. Fix winding (ensure adjacent faces have consistent edge directions)
    logging.info("    - Fixing winding...")
    try:
        trimesh.repair.fix_winding(tmesh)
    except Exception as e:
        logging.warning(f"    - Fix winding failed: {e}, continuing...")

    # 3. Fix normals (ensure normals point outward consistently)
    logging.info("    - Fixing normals...")
    try:
        trimesh.repair.fix_normals(tmesh, multibody=True)
    except Exception as e:
        logging.warning(f"    - Fix normals failed: {e}, continuing...")

    # 4. Check for broken faces and report
    try:
        broken_face_indices = trimesh.repair.broken_faces(tmesh)
        if len(broken_face_indices) > 0:
            logging.info(
                f"    - Found {len(broken_face_indices)} broken faces (marking them)"
            )
            # Mark broken faces with a distinct color if mesh has colors
            if has_colors:
                broken_color = [255, 0, 0, 255]  # Red color for broken faces
                trimesh.repair.broken_faces(tmesh, color=broken_color)
    except Exception as e:
        logging.warning(f"    - Broken faces check failed: {e}, continuing...")

    # 5. Remesh to reduce face count (simplify)
    logging.info("    - Simplifying mesh...")
    original_face_count = len(tmesh.faces)
    # target_face_count = max(int(original_face_count * 0.25), 1000)  # Reduce to 50% or min 1000 faces

    try:
        # Use trimesh's simplification
        tmesh = tmesh.simplify_quadric_decimation(remesh_percentage)
        logging.info(f"    - Reduced faces: {original_face_count} → {len(tmesh.faces)}")
    except Exception as e:
        logging.warning(f"    - Simplification failed: {e}, keeping original mesh")

    # 6. Merge close vertices to clean up geometry
    try:
        tmesh.merge_vertices()
        # Use updated methods instead of deprecated ones
        tmesh = tmesh.process(validate=False)  # Alternative to deprecated methods
        logging.info("    - Mesh cleanup completed")
    except Exception as e:
        logging.warning(f"    - Mesh cleanup failed: {e}, continuing...")

    # 7. Convert triangles to quads where possible
    # This is a simple heuristic: find pairs of adjacent triangles that share an edge
    # and could form a planar quad
    logging.info("    - Attempting to convert triangles to quads...")
    quads_created = 0
    if make_quads:
        try:
            # Limit quad detection to avoid memory issues on large meshes
            if len(tmesh.faces) < 1000000:  # Only process if less than 1M faces
                # Get face adjacency
                adjacency = tmesh.face_adjacency
                adjacency_edges = tmesh.face_adjacency_edges

                # Track which faces have been merged into quads
                merged_faces = set()
                quad_faces = []

                # Limit the number of iterations to avoid long processing times
                max_iterations = min(len(adjacency), 100000)

                for idx, ((f1, f2), edge_vertices) in enumerate(
                    zip(adjacency, adjacency_edges)
                ):
                    if idx >= max_iterations:
                        break

                    if f1 in merged_faces or f2 in merged_faces:
                        continue

                    # Get the two triangles
                    tri1 = tmesh.faces[f1]
                    tri2 = tmesh.faces[f2]

                    # Find the 4 unique vertices (2 shared, 2 unique)
                    shared_verts = set(edge_vertices)
                    unique_verts_1 = [v for v in tri1 if v not in shared_verts]
                    unique_verts_2 = [v for v in tri2 if v not in shared_verts]

                    if len(unique_verts_1) == 1 and len(unique_verts_2) == 1:
                        # Create quad from the 4 vertices in proper order
                        # Order: shared_vert1, unique_vert1, shared_vert2, unique_vert2
                        quad_verts = [
                            edge_vertices[0],
                            unique_verts_1[0],
                            edge_vertices[1],
                            unique_verts_2[0],
                        ]

                        # Check if the quad would be reasonably planar
                        try:
                            v0, v1, v2, v3 = [tmesh.vertices[i] for i in quad_verts]

                            # Simple planarity check: compare normals of the two triangular halves
                            normal1 = np.cross(v1 - v0, v2 - v0)
                            normal2 = np.cross(v2 - v0, v3 - v0)

                            if (
                                np.linalg.norm(normal1) > 0
                                and np.linalg.norm(normal2) > 0
                            ):
                                normal1 = normal1 / np.linalg.norm(normal1)
                                normal2 = normal2 / np.linalg.norm(normal2)

                                # If normals are similar (dot product close to 1), merge into quad
                                if (
                                    np.dot(normal1, normal2) > 0.95
                                ):  # ~18 degree tolerance
                                    quad_faces.append(quad_verts)
                                    merged_faces.add(f1)
                                    merged_faces.add(f2)
                                    quads_created += 1
                        except (IndexError, ValueError):
                            continue

                logging.info(
                    f"    - Created {quads_created} quads from {quads_created * 2} triangles"
                )
            else:
                logging.info(
                    f"    - Skipping quad conversion (mesh too large: {len(tmesh.faces)} faces)"
                )

            # For now, we keep the triangle mesh format since Open3D doesn't support quads directly
            # But we've logged the potential for quad conversion
            # In production, you might export to a format that supports quads (OBJ, FBX, etc.)

        except Exception as e:
            logging.warning(f"    - Quad conversion failed: {e}")

    # Return the cleaned trimesh object
    logging.info(
        f"Mesh cleaned: {len(tmesh.vertices)} vertices, {len(tmesh.faces)} triangles"
    )
    return tmesh


from scipy.spatial import KDTree


def match_grounds(
    estimated_cloud,
    gt_ground_path: str,
    device="cpu",
    xy_radius: float = 0.35,
    max_iterations: int = 20,
):
    """
    Align the estimated point cloud to match the ground plane from GT using iterative refinement.
    Only adjusts Y-coordinate (height) by finding closest points vertically.

    Like Houdini's nearpoint() - iteratively finds closest points and adjusts until convergence.

    Strategy:
    1. Load GT ground plane points (known to be ground)
    2. For each GT ground point, find closest point in estimated cloud within XZ radius
    3. Compute median Y offset
    4. Apply offset to cloud
    5. Repeat until convergence or max iterations

    Args:
        estimated_cloud: Torch tensor (N, 3) of estimated point cloud (full room)
        gt_ground_path: Path to PLY file containing GT ground plane points ONLY
        device: torch device
        xy_radius: Max XZ distance to search for nearest neighbors (meters)
        max_iterations: Maximum number of refinement iterations

    Returns:
        Adjusted point cloud with Y-offset applied
    """
    logging.info("  Matching ground planes (iterative closest point)...")

    # Load GT ground plane (only contains ground points)
    try:
        gt_trimesh = trimesh.load(gt_ground_path)
        gt_ground_points = torch.from_numpy(gt_trimesh.vertices).float().to(device)
        logging.info(f"    - Loaded GT ground: {gt_ground_points.shape[0]} points")
        logging.info(
            f"    - GT ground Y range: [{gt_ground_points[:, 1].min().item():.4f}, {gt_ground_points[:, 1].max().item():.4f}]"
        )
        logging.info(
            f"    - Initial estimated cloud Y range: [{estimated_cloud[:, 1].min().item():.4f}, {estimated_cloud[:, 1].max().item():.4f}]"
        )
    except Exception as e:
        logging.error(f"    - Failed to load GT ground plane: {e}")
        return estimated_cloud

    # Iterative refinement
    total_offset = 0.0
    adjusted_cloud = estimated_cloud.clone()

    # Move to device for faster computation
    adjusted_cloud = adjusted_cloud.to(device)
    gt_ground_points = gt_ground_points.to(device)

    # Filter predicted cloud to only bottom 25% by Y coordinate (ignore ceiling/walls)
    y_min = adjusted_cloud[:, 1].min().item()
    y_max = adjusted_cloud[:, 1].max().item()
    y_threshold = y_min + 0.15 * (y_max - y_min)  # Bottom 15% (more aggressive)

    floor_mask = adjusted_cloud[:, 1] <= y_threshold
    floor_indices = torch.where(floor_mask)[0]
    floor_cloud = adjusted_cloud[floor_mask]

    logging.info(
        f"    - Filtered to bottom 15%: {len(floor_cloud)}/{len(adjusted_cloud)} points (Y <= {y_threshold:.4f})"
    )

    # Track which floor points were matched (for coloring)
    matched_floor_indices = set()

    for iteration in range(max_iterations):
        y_distance_pairs = []

        # Compute XZ distances (horizontal plane) - use larger batches on GPU
        batch_size = (
            5000
            if device == "cuda" or (hasattr(device, "type") and device.type == "cuda")
            else 1000
        )
        for i in range(0, len(gt_ground_points), batch_size):
            batch_gt = gt_ground_points[i : i + batch_size]  # (batch, 3)

            # XZ distances only (compare against floor region only)
            batch_xz = batch_gt[:, [0, 2]]  # (batch, 2) - X and Z coords
            floor_xz = floor_cloud[:, [0, 2]]  # (N_floor, 2)

            xz_distances = torch.cdist(batch_xz, floor_xz)  # (batch, N_floor)

            # For each GT point, find closest point in XZ, measure Y-distance
            for j in range(len(batch_gt)):
                gt_point = batch_gt[j]  # (3,) full 3D coordinates

                # Find points within XZ radius
                nearby_mask = xz_distances[j] < xy_radius
                nearby_indices = torch.where(nearby_mask)[0]

                if len(nearby_indices) == 0:
                    continue

                # Among nearby points, find the one with smallest 3D distance (L2 norm)
                nearby_points = floor_cloud[nearby_indices]  # (M, 3)
                distances_3d = torch.norm(
                    nearby_points - gt_point, dim=1
                )  # L2 distance

                closest_idx = distances_3d.argmin()
                closest_point = nearby_points[closest_idx]
                closest_3d_dist = distances_3d[closest_idx].item()

                # Track matched floor point index (for coloring later)
                # Store the local floor index
                matched_floor_idx = nearby_indices[closest_idx].item()
                matched_floor_indices.add(matched_floor_idx)

                # Debug: In first iteration, also track ALL nearby points within radius
                if iteration == 0:
                    for idx in nearby_indices.cpu().tolist():
                        matched_floor_indices.add(idx)

                # Store signed Y-distance (GT - Predicted) for vertical offset
                y_distance = gt_point[1].item() - closest_point[1].item()
                y_distance_pairs.append(y_distance)

                # Debug: Store 3D distances for first iteration
                if iteration == 0 and len(y_distance_pairs) <= 10:
                    logging.debug(
                        f"      GT point {j}: Y={gt_point[1].item():.4f}, closest 3D dist={closest_3d_dist:.4f}, closest Y={closest_point[1].item():.4f}, Y-diff={y_distance:.4f}"
                    )

        if len(y_distance_pairs) == 0:
            logging.warning(
                f"    - Iteration {iteration+1}: No matches found within radius {xy_radius}m"
            )
            break

        y_distance_pairs = torch.tensor(y_distance_pairs, device=device)

        # Use mean instead of median (median is 0 when GT and predicted overlap)
        y_offset = y_distance_pairs.mean().item()
        y_median = y_distance_pairs.median().item()

        # Log distribution
        logging.debug(
            f"    - Iteration {iteration+1}: matches={len(y_distance_pairs)}, mean={y_offset:.4f}m, median={y_median:.4f}m, range=[{y_distance_pairs.min().item():.4f}, {y_distance_pairs.max().item():.4f}]"
        )

        # Check convergence
        if abs(y_offset) < 0.0000001:  # Less than 0.05mm (1/20 of a mm)
            logging.info(f"    - Converged (mean offset < 0.05mm)")
            break

        # Apply offset
        adjusted_cloud[:, 1] += y_offset
        total_offset += y_offset

        # Update floor cloud for next iteration
        floor_cloud = adjusted_cloud[floor_mask]

    logging.debug(
        f"    - Final Y range: [{adjusted_cloud[:, 1].min().item():.4f}, {adjusted_cloud[:, 1].max().item():.4f}]"
    )
    logging.info(
        f"Ground alignment completed: total offset={total_offset:.4f}m in {iteration+1} iterations"
    )

    # Move back to original device
    adjusted_cloud = adjusted_cloud.to(estimated_cloud.device)

    # Create colored point cloud: blue for matched floor points, gray for others
    colors = torch.ones(len(adjusted_cloud), 3) * 0.7  # Default gray
    matched_global_indices = floor_indices[list(matched_floor_indices)].cpu()
    colors[matched_global_indices] = torch.tensor([0.0, 0.5, 1.0])  # Blue for matched

    logging.info(
        f"    - Colored {len(matched_floor_indices)} matched floor points blue"
    )

    return adjusted_cloud, colors


def mesh_pointcloud(
    points_cloud, output_dir: str, output_name: str = "mesh", depth: int = 9, remesh_percentage: float = 0.5
):
    """
    Create a mesh from a point cloud using Poisson surface reconstruction.

    Args:
        points_cloud: Torch tensor or numpy array of shape (N, 3) with point coordinates
        output_dir: Directory to save the output mesh
        output_name: Base name for the output file (without extension)
        depth: Octree depth for Poisson reconstruction (higher = more detail, slower)
    """
    try:
        import open3d as o3d
    except ImportError:
        logging.error("Open3D not installed. Install with: pip install open3d")
        return False

    try:
        # Convert torch tensor to numpy if needed
        if torch.is_tensor(points_cloud):
            points_np = points_cloud.cpu().numpy()
        else:
            points_np = np.array(points_cloud)

        # Ensure it's the right shape and type
        points_np = np.ascontiguousarray(points_np, dtype=np.float64)

        logging.info(f"Creating mesh from point cloud...")
        logging.info(
            f"  Point cloud shape: {points_np.shape}, dtype: {points_np.dtype}"
        )

        # Create Open3D point cloud from numpy array
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)

        if len(pcd.points) == 0:
            logging.warning(f"  Skipping empty pointcloud")
            return False

        logging.info(f"  Points: {len(pcd.points)}")

        pcd.normals = o3d.utility.Vector3dVector(
            np.zeros((1, 3))
        )  # invalidate existing normals

        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)

        # Run Poisson surface reconstruction
        logging.debug(f"  Running Poisson reconstruction (depth={depth})...")


        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug
        ) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth
            )

        # Clean up mesh by removing low-density vertices
        logging.info(f"  Cleaning mesh...")
        densities = np.asarray(densities)
        vertices_to_remove = densities < np.quantile(densities, 0.025)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        mesh.remove_unreferenced_vertices()

        # Clean and remesh - returns trimesh object
        tmesh = clean_and_remesh(mesh, remesh_percentage)

        if tmesh is None:
            logging.error("  ❌ Failed to clean and remesh")
            return False

        logging.info(
            f"  Final mesh: {len(tmesh.vertices)} vertices, {len(tmesh.faces)} triangles"
        )

        # Save mesh directly with trimesh
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_name)

        # Trimesh can save directly to glb
        tmesh.export(output_path, file_type="glb")
        logging.info(f"Saved to {output_path}")

        return True

    except Exception as e:
        logging.error(f"Failed: {e}")
        return False


def mesh_background(config: dict = None, pc_file: str = ""):
    if config is None:
        raise ValueError("Config dictionary is required")

    if not os.path.exists(pc_file):
        logging.error(
            f"Point cloud file not found: {pc_file}, skipping background meshing"
        )
        return

    vggt_scale = config.get("vggt_scene_scale", 2.0)
    output_dir = config.get("out_pc_meshed", "../output/pointclouds/meshed")
    os.makedirs(output_dir, exist_ok=True)

    logging.debug(f"Input file: {pc_file}")
    logging.debug(f"Output directory: {output_dir}")
    logging.debug(f"VGGT scene scale: {vggt_scale}")
    logging.debug("=" * 80)

    # Prepare cloud from PLY file with single transformation
    cloud = set_vggt_cloud(pc_file, vggt_scene_scale=vggt_scale, device="cpu")

    # Align to ground plane (use CUDA for speed)
    device = config.get("device", "cpu")

    gt_ground = os.path.join(config.get("output_ply"), "PLANE_SAMPLED.ply")

    if os.path.exists(gt_ground):
        cloud, colors = match_grounds(
            cloud,
            gt_ground,
            device=device,
            xy_radius=config.get("point_search_radius", 0.05),
            max_iterations=config.get("max_ground_matching_iterations", 20),
        )
    else:
        logging.error(
            f"GT ground file not found: {gt_ground}, skipping ground alignment"
        )
        colors = None

    # Save aligned point cloud for visualization
    output_ply_path = os.path.join(output_dir, "ground_aligned.ply")

    # Convert to trimesh and save with colors
    cloud_np = cloud.cpu().numpy()

    if colors is not None:
        colors_np = colors.cpu().numpy()
        cloud_trimesh = trimesh.PointCloud(vertices=cloud_np, colors=colors_np)
    else:
        cloud_trimesh = trimesh.PointCloud(vertices=cloud_np)

    cloud_trimesh.export(output_ply_path)
    logging.info(f"Saved aligned point cloud to: {output_ply_path}")

    # Comment out meshing for now
    mesh_pointcloud(
        cloud,
        output_dir,
        output_name="ground_aligned.glb",
        depth=config.get("background_mesh_depth", 3),
        remesh_percentage=config.get("background_remesh_percentage", 0.5)
    )
    logging.debug("=" * 80)
