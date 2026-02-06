# modul import, check for pytorch3d and install if necessary

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
from pytorch3d.io import IO

from scipy import io

# logging.info curred working directory
import os, sys
import logging

logging.info("Current working directory:", os.getcwd())
# append src/scene_reconstruction/source to sys.path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.getcwd(), "src/scene_reconstruction/source"))
)

from source.utils_SR.diff_utils import (
    # normalized_to_camera_space,
    # dice_loss,
    # extract_camera_from_json,
    # camera_to_world_space,
    # regularize_depth_map,
    clean_mesh,
    sample_mesh_points,
    # depth_from_image,
    visualize_plane_and_axes,
    save_glb_mesh,
    visualize_pointclouds,
)

from source.utils_SR.cam_utils import (
    calibrate_cameras,
)
from source.utils_SR.render_utils import (
    initialize_renderer,
    make_pointcloud_renderer,
)
from source.utils_SR.pc_utils import get_model_vggt_cloud

from source.diff_model_planar import Model as PlanarModel
from source.diff_model import Model as RegularModel
import os
import sys

import torch
from cv2 import resize

import trimesh
import torch
import numpy as np
from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from skimage import img_as_ubyte


sys.path.insert(0, "../")
from utils.global_utils import (
    clear_output_directory,
    calculate_iou,
    save_img_to_temp,
    save_point_cloud,
)


import logging
import warnings


from pytorch3d.loss import chamfer_distance, point_mesh_face_distance
from pytorch3d.transforms import euler_angles_to_matrix, Transform3d
import torch.nn.functional as F

need_pytorch3d = False

try:
    import pytorch3d
except ModuleNotFoundError:
    need_pytorch3d = True

if need_pytorch3d:
    if torch.__version__.startswith("2.2.") and sys.platform.startswith("linux"):
        # We try to install PyTorch3D via a released wheel.
        pyt_version_str = torch.__version__.split("+")[0].replace(".", "")
        version_str = "".join(
            [
                f"py3{sys.version_info.minor}_cu",
                torch.version.cuda.replace(".", ""),
                f"_pyt{pyt_version_str}",
            ]
        )
        #!pip install fvcore iopath
        #!pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html
    else:
        # We try to install PyTorch3D from source.
        logging.info("Installing PyTorch3D from source. This may take a while...")
        # %pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'


def get_plane_transforms(plane_normal: torch.Tensor, plane_point: torch.Tensor):
    """
    Computes the world_to_plane and plane_to_world transformation matrices.
    Creates a coordinate frame where:
    - Plane-local Y-axis = plane normal (objects move up/down perpendicular to plane)
    - Plane-local X, Z axes = tangent to plane (objects slide along these)
    """
    device = plane_normal.device

    # 1. Plane-local Y-axis is the plane normal (perpendicular to surface)
    y_axis = F.normalize(plane_normal.view(1, 3), p=2, dim=1)

    # 2. Choose a reference direction for X-axis (avoid parallel to normal)
    ref = torch.tensor([[1.0, 0.0, 0.0]], device=device)  # Try world X first
    if torch.allclose(torch.abs(y_axis.squeeze()), torch.abs(ref.squeeze()), atol=0.9):
        ref = torch.tensor(
            [[0.0, 0.0, 1.0]], device=device
        )  # Use world Z if Y is close to X

    # Cross product to get first tangent direction
    z_axis = F.normalize(torch.cross(y_axis, ref, dim=1), p=2, dim=1)

    # 3. Second tangent direction (complete right-handed frame)
    x_axis = F.normalize(torch.cross(y_axis, z_axis, dim=1), p=2, dim=1)

    # 4. Create the rotation matrix from the new axes
    # For plane-to-world transform, we need the rotation matrix where:
    # - Rows show where world axes go in plane space (world-to-plane)
    # - Columns show where plane axes go in world space (plane-to-world)
    # PyTorch3D .rotate() expects the matrix that transforms points FROM source TO target
    # We want: plane coords -> world coords, so we need columns = world coords of plane axes
    R_plane_to_world_cols = torch.cat([x_axis.T, y_axis.T, z_axis.T], dim=1)  # (3, 3)

    # But PyTorch3D might expect rows instead of columns!
    # Let's try the transpose
    R_plane_to_world = (
        R_plane_to_world_cols.T
    )  # Transpose: now rows are where plane axes point in world

    logging.debug(f"Plane coordinate system (in world coords):")
    logging.debug(f"  Plane-X axis (tangent): {x_axis.squeeze().cpu().numpy()}")
    logging.debug(f"  Plane-Y axis (normal):  {y_axis.squeeze().cpu().numpy()}")
    logging.debug(f"  Plane-Z axis (tangent): {z_axis.squeeze().cpu().numpy()}")

    # 5. Create the full 4x4 transformation matrices
    # Ensure plane_point is a 1D tensor of shape (3,)
    if plane_point.dim() > 1:
        plane_point_1d = plane_point.squeeze()
    else:
        plane_point_1d = plane_point

    # Test the rotation first (no translation)
    T_rotate_only = Transform3d(device=device).rotate(R_plane_to_world)
    test_y_axis = torch.tensor(
        [[0.0, 1.0, 0.0]], device=device
    )  # Plane-Y axis (normal direction)
    rotated_y = T_rotate_only.transform_points(test_y_axis.unsqueeze(0)).squeeze()
    logging.debug(
        f"Verification: Rotating plane-Y (0,1,0) gives: {rotated_y.cpu().numpy()}"
    )
    logging.debug(
        f"Expected to match plane normal: {plane_normal.squeeze().cpu().numpy()}"
    )

    # Create rotation transform
    T_rotate = Transform3d(device=device).rotate(R_plane_to_world)

    # Create translation transform using individual components
    # translate() expects x, y, z as separate arguments or a tensor
    T_translate = Transform3d(device=device).translate(
        plane_point_1d[0].item(), plane_point_1d[1].item(), plane_point_1d[2].item()
    )

    # Compose: first rotate, then translate
    T_plane_to_world = T_rotate.compose(T_translate)

    # The inverse transform maps from world to our simplified plane space
    T_world_to_plane = T_plane_to_world.inverse()

    return T_plane_to_world, T_world_to_plane


def find_best_initial_yaw(
    mesh_verts: torch.Tensor,
    target_points: torch.Tensor,
    num_angles: int = 18,
    use_chamfer: bool = True,
    faces: torch.Tensor = None,
    debug_save: bool = False,
    debug_dir: str | None = None,
    debug_prefix: str | None = None,
    centroid_method: str = "bbox",
) -> torch.Tensor:
    """
    Performs a grid search to find the best initial yaw (Y-axis) rotation
    by comparing shapes at the origin.

    Args:
        mesh_verts: Mesh vertices (N, 3)
        target_points: Target point cloud (M, 3)
        num_angles: Number of angles to test
        use_chamfer: If True, use chamfer_distance (batched, faster).
                     If False, use point_mesh_face_distance (loop-based, slower but more accurate)
    """
    logging.info(f"Searching for best initial yaw across {num_angles} angles...")
    logging.info(
        f"Using {'chamfer_distance (batched)' if use_chamfer else 'point_mesh_face_distance (loop-based)'} | center: {centroid_method}"
    )
    device = mesh_verts.device

    # Center both point clouds before comparison using requested method
    def _compute_center(pts: torch.Tensor, method: str) -> torch.Tensor:
        m = (method or "mean").lower()
        if m == "bbox":
            mins = pts.min(dim=0).values
            maxs = pts.max(dim=0).values
            return (mins + maxs) / 2.0
        if m == "median":
            return pts.median(dim=0).values
        return pts.mean(dim=0)

    mesh_centroid = _compute_center(mesh_verts, centroid_method)
    centered_mesh_verts = mesh_verts - mesh_centroid

    target_centroid = _compute_center(target_points, centroid_method)
    centered_target_points = target_points - target_centroid

    # Create a batch of candidate yaw angles
    candidate_yaws = torch.linspace(0, 2 * 3.14159, num_angles, device=device)
    euler_angles = torch.zeros(num_angles, 3, device=device)
    euler_angles[:, 1] = candidate_yaws
    R_candidates = euler_angles_to_matrix(euler_angles, convention="XYZ")

    # Batched versions for efficient computation
    verts_batched = centered_mesh_verts.unsqueeze(0).expand(num_angles, -1, -1)

    # Perform rotation AT THE ORIGIN
    rotated_verts_batched = torch.bmm(verts_batched, R_candidates.transpose(1, 2))

    # Optional debug dump of point clouds for visual inspection
    if debug_save:
        try:
            out_root = debug_dir if debug_dir is not None else "../output"
            prefix = debug_prefix if debug_prefix is not None else "model"
            dump_dir = os.path.join(out_root, "rot_grid_debug", prefix)
            os.makedirs(dump_dir, exist_ok=True)

            # Save centered target and centered mesh (unrotated)
            save_point_cloud(
                centered_target_points,
                os.path.join(dump_dir, "target_centered.ply"),
                blender_readable=False,
            )
            save_point_cloud(
                centered_mesh_verts,
                os.path.join(dump_dir, "mesh_centered.ply"),
                blender_readable=False,
            )

            # Save a rotated version for each candidate angle
            for i in range(num_angles):
                angle_deg = candidate_yaws[i].item() * 180.0 / 3.14159
                rot_i = rotated_verts_batched[i]
                save_point_cloud(
                    rot_i,
                    os.path.join(dump_dir, f"mesh_rot_{angle_deg:.1f}.ply"),
                    blender_readable=False,
                )
        except Exception as e:
            logging.warning(f"Failed to dump rotation grid debug pointclouds: {e}")

    if use_chamfer:
        # Chamfer distance - works with batches
        target_points_batched = centered_target_points.unsqueeze(0).expand(
            num_angles, -1, -1
        )
        loss, _ = chamfer_distance(
            rotated_verts_batched,
            target_points_batched,
            batch_reduction=None,  # Returns tensor of shape (num_angles,)
        )
    else:
        # Point-mesh-face distance - needs loop (doesn't support batching)
        if faces is None:
            raise ValueError("Faces must be provided when use_chamfer=False")

        # Convert faces list to tensor if needed
        if isinstance(faces, list):
            # faces_list returns a list of tensors, take the first one
            faces_tensor = faces[0] if len(faces) > 0 else None
            if faces_tensor is None:
                raise ValueError("Empty faces list provided")
        else:
            faces_tensor = faces

        centered_target_pc = Pointclouds(points=[centered_target_points])
        losses = []
        for i in range(num_angles):
            rotated_verts = rotated_verts_batched[i]  # (N, 3)
            # Create Meshes object with verts as list and faces as list
            rotated_mesh = Meshes(verts=[rotated_verts], faces=[faces_tensor])
            loss_i = point_mesh_face_distance(rotated_mesh, centered_target_pc)
            losses.append(loss_i)
        loss = torch.stack(losses).squeeze()  # (num_angles,)

    # Find best angle
    best_angle_idx = torch.argmin(loss)
    best_R = R_candidates[best_angle_idx]

    best_angle_deg = candidate_yaws[best_angle_idx].item() * 180 / 3.14159
    logging.info(
        f"✅ Best initial angle found: {best_angle_deg:.2f} degrees with loss {loss[best_angle_idx]:.6f}"
    )

    # Save the best rotation as well if debugging
    if debug_save:
        try:
            out_root = debug_dir if debug_dir is not None else "../output"
            prefix = debug_prefix if debug_prefix is not None else "model"
            dump_dir = os.path.join(out_root, "rot_grid_debug", prefix)
            os.makedirs(dump_dir, exist_ok=True)

            best_rot = rotated_verts_batched[best_angle_idx]
            save_point_cloud(
                best_rot,
                os.path.join(dump_dir, f"mesh_rot_best_{best_angle_deg:.1f}.ply"),
                blender_readable=False,
            )
        except Exception as e:
            logging.warning(f"Failed to dump best rotation pointcloud: {e}")

    return best_R


def get_oriented_bounding_box_2d_up(points: torch.Tensor):
    """
    Computes the OBB for a point cloud while keeping the Y-axis fixed as "up".
    This finds the rotation around the Y-axis and the extents in that new frame.

    Args:
        points (torch.Tensor): Point cloud of shape (N, 3).

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - centroid (3,): The 3D center of the OBB.
            - rotation (3, 3): The 3D rotation matrix (only rotating around Y-axis).
            - extents (3,): The dimensions of the OBB.
    """
    if points.numel() == 0:
        return torch.zeros(3), torch.eye(3), torch.zeros(3)

    # 1. Compute the 3D centroid
    centroid = points.mean(dim=0)

    # 2. Perform 2D PCA on the XZ plane to find the ground-plane rotation
    points_2d = points[:, [0, 2]]
    centered_points_2d = points_2d - points_2d.mean(dim=0, keepdim=True)
    covariance = torch.matmul(centered_points_2d.T, centered_points_2d) / len(points_2d)
    _, _, V = torch.linalg.svd(covariance)  # V is the 2x2 rotation matrix

    # 3. "Lift" the 2x2 rotation into a 3x3 rotation matrix for the Y-axis
    # This matrix only rotates around Y, leaving the up-axis untouched.
    rotation = torch.eye(3, device=points.device, dtype=points.dtype)
    rotation[0, 0] = V[0, 0]
    rotation[0, 2] = V[0, 1]
    rotation[2, 0] = V[1, 0]
    rotation[2, 2] = V[1, 1]

    # 4. Project the original 3D points onto these new axes to find the extents
    projected_points = torch.matmul(points - centroid, rotation)
    min_coords, _ = torch.min(projected_points, dim=0)
    max_coords, _ = torch.max(projected_points, dim=0)
    extents = max_coords - min_coords

    return centroid, rotation, extents


# test for outlier point in pccloud


def rotate_around_y(chair_verts, angle_degrees, device):
    # Generate a random rotation angle about the Y-axis
    angle = angle_degrees * (3.14159265 / 180.0)  # Convert to radians
    # Create the Y-axis rotation matrix
    rotation_matrix = torch.tensor(
        [
            [torch.cos(torch.tensor(angle)), 0, torch.sin(torch.tensor(angle))],
            [0, 1, 0],
            [-torch.sin(torch.tensor(angle)), 0, torch.cos(torch.tensor(angle))],
        ],
        device=device,
        dtype=torch.float32,
    )
    # Rotate the vertices: since chair_verts_translated is (1, V, 3) we can use matrix multiplication
    chair_verts = torch.matmul(chair_verts, rotation_matrix.T)

    return chair_verts


def fit_plane_svd(points):
    """
    Fit a plane using SVD (Total Least Squares).
    Minimizes perpendicular distances - optimal for planes.

    Args:
        points: (N, 3) tensor of 3D points

    Returns:
        plane_normal: (3,) tensor - unit normal vector
        plane_centroid: (1, 3) tensor - point on the plane
    """
    centroid = points.mean(dim=0)
    centered = points - centroid

    # SVD: columns of V are principal components
    _, _, V = torch.linalg.svd(centered.T @ centered)

    # Normal is the direction with smallest variance (last column)
    normal = V[:, -1]

    # Ensure normal points UP (positive Y)
    if normal[1] < 0:
        normal = -normal

    return normal, centroid.unsqueeze(0)


def fit_plane_ransac_refined(points, iterations=2000, threshold=0.05):
    """
    RANSAC for outlier rejection + SVD refinement on inliers.
    Combines robustness of RANSAC with accuracy of least squares.

    Args:
        points: (N, 3) tensor of 3D points
        iterations: number of RANSAC iterations
        threshold: inlier distance threshold in meters

    Returns:
        plane_normal: (3,) tensor - unit normal vector
        plane_point: (1, 3) tensor - centroid of inliers
        inlier_mask: (N,) boolean tensor - which points are inliers
    """
    best_inliers = 0
    best_inlier_mask = None

    for _ in range(iterations):
        # Sample 3 random points
        idx = torch.randperm(len(points))[:3]
        pts = points[idx]

        # Compute plane from 3 points
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        normal = torch.cross(v1, v2)
        if torch.norm(normal) < 1e-6:
            continue  # Skip degenerate cases
        normal = normal / torch.norm(normal)

        # Count inliers
        distances = torch.abs((points - pts[0]) @ normal)
        inlier_mask = distances < threshold
        num_inliers = inlier_mask.sum().item()

        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_inlier_mask = inlier_mask

    # Refine using SVD on inliers only
    inlier_points = points[best_inlier_mask]
    normal, centroid = fit_plane_svd(inlier_points)

    return normal, centroid, best_inlier_mask


def extract_and_fit_floor_plane(
    floor_mask: np.ndarray,
    vggt_cloud_path: str,
    cameras,
    point_cloud_folder: str,
    device: torch.device,
):
    """
    Extract floor pointcloud from VGGT using floor mask and fit optimal plane.

    Compares three plane fitting methods:
    1. SVD (Total Least Squares) - accurate for clean data
    2. RANSAC+SVD - robust to outliers
    3. Axis-aligned - simple, forces alignment with min-variance axis

    Args:
        model_image_path: Path to the model mask image
        vggt_cloud_path: Path to VGGT pointcloud PLY file
        cameras: PyTorch3D camera object
        image_size: Width of rendered images
        target_height: Height of rendered images
        point_cloud_folder: Directory to save diagnostic PLY files
        device: Torch device (CPU or CUDA)

    Returns:
        plane_normal: (3,) tensor - unit normal vector of fitted plane
        plane_point: (1, 3) tensor - point on the plane
        plane_to_world: Transform3d from plane-local to world coordinates
        world_to_plane: Transform3d from world to plane-local coordinates
        floor_pointcloud: (N, 3) tensor - extracted floor points
    """
    logging.debug("=" * 80)
    logging.debug("EXTRACTING GROUND PLANE FROM FLOOR MASK")
    logging.debug("=" * 80)

    # Extract floor pointcloud from VGGT using the floor mask
    floor_pointcloud = get_model_vggt_cloud(
        mask=floor_mask, vggt_cloud_path=vggt_cloud_path, cameras=cameras, device=device
    )

    logging.debug(f"Floor pointcloud extracted: {floor_pointcloud.shape[0]} points")

    # Analyze floor orientation using PCA
    logging.debug("=" * 80)
    logging.debug("FLOOR PLANARITY ANALYSIS (PCA)")
    logging.debug("=" * 80)

    # Center the floor pointcloud
    floor_centered = floor_pointcloud - floor_pointcloud.mean(dim=0, keepdim=True)

    # Compute covariance matrix and perform PCA
    cov_matrix = (floor_centered.T @ floor_centered) / (floor_centered.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    # Sort by eigenvalue (descending)
    idx = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # The eigenvector with smallest eigenvalue is the normal
    pca_normal = eigenvectors[:, 2]  # Smallest eigenvalue -> normal direction

    # Ensure normal points upward (positive Y component for upward-facing floor)
    if pca_normal[1] < 0:
        pca_normal = -pca_normal

    logging.debug(f"PCA Eigenvalues (variance along principal axes):")
    logging.debug(f"  PC1 (largest):  {eigenvalues[0]:.6f}  <- main spread direction")
    logging.debug(f"  PC2 (medium):   {eigenvalues[1]:.6f}  <- secondary spread")
    logging.debug(
        f"  PC3 (smallest): {eigenvalues[2]:.6f}  <- perpendicular to plane (flatness)"
    )
    logging.debug(
        f"Planarity ratio: {eigenvalues[2]/eigenvalues[0]:.6f}  (lower = flatter)"
    )
    logging.debug(f"PCA normal (from smallest eigenvector): {pca_normal.cpu().numpy()}")
    logging.debug("=" * 80)

    floor_min = floor_pointcloud.min(dim=0).values
    floor_max = floor_pointcloud.max(dim=0).values
    floor_mean = floor_pointcloud.mean(dim=0)
    floor_std = floor_pointcloud.std(dim=0)

    logging.debug(
        f"X range: [{floor_min[0]:.4f}, {floor_max[0]:.4f}] mean={floor_mean[0]:.4f} std={floor_std[0]:.4f}"
    )
    logging.debug(
        f"Y range: [{floor_min[1]:.4f}, {floor_max[1]:.4f}] mean={floor_mean[1]:.4f} std={floor_std[1]:.4f}"
    )
    logging.debug(
        f"Z range: [{floor_min[2]:.4f}, {floor_max[2]:.4f}] mean={floor_mean[2]:.4f} std={floor_std[2]:.4f}"
    )

    # Check which axis has smallest variance (should be the floor normal)
    logging.debug(
        f"Variance per axis: X={floor_std[0]:.4f}, Y={floor_std[1]:.4f}, Z={floor_std[2]:.4f}"
    )
    min_variance_axis = torch.argmin(floor_std)
    axis_names = ["X", "Y", "Z"]
    logging.debug(
        f"Smallest variance: {axis_names[min_variance_axis]}-axis (std={floor_std[min_variance_axis]:.4f})"
    )
    logging.debug(f"Floor is a plane perpendicular to {axis_names[min_variance_axis]}")

    # Create a simple plane aligned with the min-variance axis
    forced_normal = torch.zeros(3, device=device)
    forced_normal[min_variance_axis] = 1.0  # Start with positive direction

    # Normal should point AWAY from floor surface (upward)
    if min_variance_axis == 1:  # Y-axis
        if floor_mean[1] < 0:  # Floor below origin
            forced_normal[1] = 1.0  # Normal points up (positive Y)
        else:  # Floor above origin
            forced_normal[1] = -1.0  # Normal points down (negative Y)
        logging.debug(
            f"Floor at Y={floor_mean[1]:.4f}, normal points {'up (+Y)' if forced_normal[1] > 0 else 'down (-Y)'}"
        )
        logging.debug(f"Forcing normal to: {forced_normal.cpu().numpy()}")

    # Compare plane fitting methods
    logging.debug("=" * 80)
    logging.debug("COMPARING PLANE FITTING METHODS")
    logging.debug("=" * 80)

    # Method 1: Pure SVD (least squares)
    svd_normal, svd_point = fit_plane_svd(floor_pointcloud)
    svd_distances = torch.abs((floor_pointcloud - svd_point) @ svd_normal)
    svd_rmse = torch.sqrt((svd_distances**2).mean()).item()

    logging.debug(f"[SVD] Total Least Squares:")
    logging.debug(f"  Normal: {svd_normal.cpu().numpy()}")
    logging.debug(f"  Point: {svd_point.squeeze().cpu().numpy()}")
    logging.debug(f"  RMSE: {svd_rmse:.6f}")
    logging.debug(f"  Max error: {svd_distances.max().item():.6f}")

    # Method 2: RANSAC + SVD refinement (robust to outliers)
    ransac_normal, ransac_point, inlier_mask = fit_plane_ransac_refined(
        floor_pointcloud, iterations=2000, threshold=0.05  # 5cm tolerance
    )
    ransac_distances = torch.abs((floor_pointcloud - ransac_point) @ ransac_normal)
    ransac_rmse = torch.sqrt((ransac_distances**2).mean()).item()
    inlier_ratio = inlier_mask.sum().item() / len(floor_pointcloud)

    logging.debug(f"[RANSAC+SVD] Outlier-robust refinement:")
    logging.debug(f"  Normal: {ransac_normal.cpu().numpy()}")
    logging.debug(f"  Point: {ransac_point.squeeze().cpu().numpy()}")
    logging.debug(f"  RMSE: {ransac_rmse:.6f}")
    logging.debug(
        f"  Inliers: {inlier_mask.sum().item()}/{len(floor_pointcloud)} ({100*inlier_ratio:.1f}%)"
    )

    # Method 3: Axis-aligned (force alignment with min-variance axis)
    forced_point = floor_mean.unsqueeze(0)  # Use centroid
    forced_distances = torch.abs((floor_pointcloud - forced_point) @ forced_normal)
    forced_rmse = torch.sqrt((forced_distances**2).mean()).item()

    logging.debug(f"[AXIS-ALIGNED] Forced alignment with min-variance axis:")
    logging.debug(f"  Normal: {forced_normal.cpu().numpy()}")
    logging.debug(f"  Point: {forced_point.squeeze().cpu().numpy()}")
    logging.debug(f"  RMSE: {forced_rmse:.6f}")
    logging.debug(f"  Max error: {forced_distances.max().item():.6f}")

    # Compare all methods
    logging.debug(f"COMPARISON (lower RMSE = better fit):")
    logging.debug(f"  SVD RMSE:          {svd_rmse:.6f}")
    logging.debug(f"  RANSAC+SVD RMSE:   {ransac_rmse:.6f}")
    logging.debug(f"  AXIS-ALIGNED RMSE: {forced_rmse:.6f}")

    # Choose the best method based on data quality
    # IMPORTANT: For very flat floors (planarity < 0.001), use PCA normal
    # SVD can give wrong normals due to numerical issues with nearly-planar data
    planarity_ratio = eigenvalues[2] / eigenvalues[0]

    if planarity_ratio < 0.001:
        # Floor is extremely flat - use PCA normal (most accurate)
        logging.debug(
            f"Using PCA plane fitting (planarity ratio {planarity_ratio:.6f} < 0.001)"
        )
        plane_normal = pca_normal
        plane_point = floor_mean.unsqueeze(0)
        inliers = len(floor_pointcloud)
    elif inlier_ratio < 0.85:
        # Significant outliers detected - use RANSAC for robustness
        logging.debug("Using RANSAC+SVD plane fitting (outliers detected)")
        plane_normal, plane_point = ransac_normal, ransac_point
        inliers = inlier_mask.sum().item()
    else:
        # Use SVD for moderately planar data
        logging.debug("Using SVD plane fitting (captures actual VGGT floor tilt)")
        plane_normal, plane_point = svd_normal, svd_point
        inliers = len(floor_pointcloud)

    logging.debug(
        f"Final plane - Normal: {plane_normal.cpu().numpy()}, Point: {plane_point.squeeze().cpu().numpy()}"
    )
    logging.debug(
        f"Coverage: {inliers}/{len(floor_pointcloud)} ({100*inliers/len(floor_pointcloud):.1f}%)"
    )

    # Save floor pointcloud for debugging
    floor_ply_path = os.path.join(point_cloud_folder, "FLOOR.ply")
    save_point_cloud(floor_pointcloud, floor_ply_path, blender_readable=True)
    logging.debug(f"Floor pointcloud saved to: {floor_ply_path}")

    # Save color-coded residuals for visual debugging
    final_distances = torch.abs((floor_pointcloud - plane_point) @ plane_normal)
    max_dist = final_distances.max().item()
    mean_dist = final_distances.mean().item()
    median_dist = final_distances.median().item()

    logging.debug(f"Floor fit quality:")
    logging.debug(f"  Mean distance to plane: {mean_dist:.4f}m")
    logging.debug(f"  Median distance: {median_dist:.4f}m")
    logging.debug(f"  Max distance: {max_dist:.4f}m")
    logging.debug(f"  RMSE: {svd_rmse:.6f}m")

    # Create color map: green (close) -> yellow -> red (far)
    colors = torch.zeros((len(floor_pointcloud), 3), device=device)
    normalized_dist = (final_distances / max_dist).clamp(0, 1)
    colors[:, 0] = normalized_dist  # Red channel increases with error
    colors[:, 1] = 1 - normalized_dist  # Green channel decreases with error

    # Save as PLY with RGB colors
    floor_colored = torch.cat([floor_pointcloud, colors], dim=1)  # (N, 6)
    floor_colored_np = floor_colored.cpu().numpy()

    # Create trimesh point cloud with colors
    floor_pc_colored = trimesh.PointCloud(
        vertices=floor_colored_np[:, :3],
        colors=(floor_colored_np[:, 3:] * 255).astype(np.uint8),
    )
    floor_pc_colored.export(os.path.join(point_cloud_folder, "FLOOR_RESIDUALS.ply"))
    logging.debug(
        f"Color-coded residuals saved (green=good, red=bad, max_error={max_dist:.4f}m)"
    )

    # Get plane transforms
    plane_to_world, world_to_plane = get_plane_transforms(plane_normal, plane_point)

    # Sample points on the fitted plane and save for visualization
    grid_size = 100  # 100x100 points
    grid_extent = 3.0  # +/- 3 meters in each direction
    u = torch.linspace(-grid_extent, grid_extent, grid_size, device=device)
    v = torch.linspace(-grid_extent, grid_extent, grid_size, device=device)
    uu, vv = torch.meshgrid(u, v, indexing="ij")

    # Create points in plane coordinates (x, 0, z) = on the floor surface
    plane_points_local = torch.stack(
        [
            uu.flatten(),  # plane-X (tangent to floor)
            torch.zeros_like(uu.flatten()),  # plane-Y = 0 (ON the floor)
            vv.flatten(),  # plane-Z (tangent to floor)
        ],
        dim=1,
    )  # (N, 3)

    # Transform to world coordinates
    plane_points_world = plane_to_world.transform_points(
        plane_points_local.unsqueeze(0)
    ).squeeze(0)

    # Verify the sampled points actually lie on the fitted plane
    verification_distances = torch.abs(
        (plane_points_world - plane_point) @ plane_normal
    )
    max_verification_error = verification_distances.max().item()
    mean_verification_error = verification_distances.mean().item()

    logging.debug(f"Sampled plane verification:")
    logging.debug(f"  Mean distance from fitted plane: {mean_verification_error:.6f}m")
    logging.debug(f"  Max distance from fitted plane: {max_verification_error:.6f}m")
    if max_verification_error > 0.001:
        logging.warning(
            f"⚠️ Sampled plane points don't lie on fitted plane! Max error: {max_verification_error:.6f}m"
        )
    else:
        logging.debug(f"✅ Sampled plane correctly lies on fitted plane (error < 1mm)")

    # Save sampled plane points
    plane_ply_path = os.path.join(point_cloud_folder, "PLANE_SAMPLED.ply")
    save_point_cloud(plane_points_world, plane_ply_path, blender_readable=False)
    logging.debug(f"Saved visualization files:")
    logging.debug(f"  FLOOR.ply - Original {len(floor_pointcloud)} VGGT floor points")
    logging.debug(
        f"  FLOOR_RESIDUALS.ply - Same points colored by distance from fitted plane"
    )
    logging.debug(
        f"  PLANE_SAMPLED.ply - {grid_size}x{grid_size}={len(plane_points_world)} grid points on fitted plane"
    )
    logging.debug(
        f"Note: FLOOR_RESIDUALS shows real scattered data, PLANE_SAMPLED shows ideal mathematical plane"
    )

    return plane_normal, plane_point, plane_to_world, world_to_plane, floor_pointcloud


##################
# Helper
##################


def clear_pose_matching_outputDir(config: dict, iteration: int = 0):
    output_folder = config.get("output", "../output")

    # Create output folders if they do not exist
    final_glb_path = config.get("glb_output_folder", "../output/glb")
    if not os.path.exists(os.path.dirname(final_glb_path)):
        os.makedirs(os.path.dirname(final_glb_path))

    # Save the final rendered image
    gif_folder = os.path.join(output_folder, "gif")
    if not os.path.exists(gif_folder):
        os.makedirs(gif_folder)

    # Create figures folder
    figures_folder = os.path.join(output_folder, "figures")
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)

    # if first iteration, clear the output folders
    if iteration == 0:
        # clear output directory if it exists
        clear_output_directory(final_glb_path)
        # Clear gif folder if it exists
        clear_output_directory(gif_folder)
        # Clear figures folder if it exists
        clear_output_directory(figures_folder)

    return final_glb_path, gif_folder, figures_folder


def add_frames_gif(
    step, step_size, phong_renderer, frames, model=None, background=None
):
    """
    Adds frames to the GIF with step size

    Add background if wanted.
    """

    if step % step_size == 0:
        if frames is None:
            frames = []

        # Assert error when model is missing
        assert model is not None, "Model is not provided"

        # Render a full Phong scene for visualization
        scene_render = phong_renderer(meshes_world=model)
        scene_tmp = scene_render.detach().cpu().numpy().squeeze()

        # Convert the rendered image to uint8 (0-255)
        rendered_image_np = img_as_ubyte(scene_tmp)
        alpha = rendered_image_np[..., 3:4].astype(np.float32) / 255.0

        if background is None:  # Make background transparent
            composite = rendered_image_np[..., :3].astype(np.float32)  # * alpha
        else:
            # Blend the rendered image and the background
            composite = rendered_image_np[..., :3].astype(
                np.float32
            ) * alpha + background * (1 - alpha)

        composite = composite.astype(np.uint8)

        # Validate composite before appending
        if composite.ndim == 3 and composite.shape[-1] in [
            3,
            4,
        ]:  # Ensure it's an RGB(A) image
            frames.append(composite)
        else:
            raise ValueError(f"Invalid frame shape: {composite.shape}")

    return frames


#########################################################################################################################################
# Start of main Script
#########################################################################################################################################


def pose_matching(config, model_name, iteration, device_id):
    # When CUDA_VISIBLE_DEVICES is set, there's only one visible GPU (index 0)
    if torch.cuda.is_available():
        if device_id is None:
            device_id = 0
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
        logging.debug(f"Using GPU {device_id} for model {model_name}")
    else:
        device = torch.device("cpu")
        logging.debug("Using CPU")

    # set Logging levels
    level = getattr(logging, str(config.get("logging", "INFO")).upper(), logging.debug)
    logging.basicConfig(level=level, format="%(message)s", force=True)

    # clear output directory ####################################################
    final_glb_path, gif_folder, figures_folder = clear_pose_matching_outputDir(
        config, iteration
    )

    # Load the obj and ignore the textures and materials.
    # load objects ####################################################
    io = IO()
    io.register_meshes_format(MeshGlbFormat())
    # Load files

    #logging.info("IO model : ", io)

    glb_path = config.get("output_folder_hy", None)
    if glb_path is None:
        raise ValueError("GLB path not provided in config.")

    # # Use textured model from Hunyuan3D-2.1 (it's OBJ format with .glb extension)
    # textured_path = os.path.join(glb_path, model_name, model_name + ".glb")

    # # check if textured file exists
    # if not os.path.exists(textured_path):
    #     raise FileNotFoundError(f"Textured model not found at: {textured_path}")
    glb_path = os.path.join(glb_path, model_name, model_name + ".glb")

    # check if glb exists
    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"GLB file not found at: {glb_path}")

    logging.info("Loading GLB model from:", glb_path)

    load_object = io.load_mesh(glb_path, load_textures=True, device=device)

    # clean mesh
    glb_object = clean_mesh(load_object)

    # logging debugging statements
    logging.debug("GLB Object Loaded:")
    logging.debug(glb_object)

    logging.debug("Verts shape: %s", glb_object.verts_padded().shape)
    logging.debug("Faces shape: %s", glb_object.faces_padded().shape)

    verts = glb_object.verts_padded()[0]
    logging.debug("Min: %s", verts.min(dim=0).values)
    logging.debug("Max: %s", verts.max(dim=0).values)
    verts_centroid = verts.mean(dim=0)
    logging.debug("Mean: %s", verts_centroid)

    # set global variables ####################################################
    image_size = config.get("image_size_DR", 512)

    sigma = float(config.get("sigma", 5e-5))  # sigma for SoftSilhouetteShader
    gamma = float(config.get("gamma", 5e-5))  # gamma for HardPhongShader

    # set manual seed
    torch.manual_seed(config.get("seed", 12345))

    # load image ####################################################

    model_image_folder = config.get("full_size", None)
    model_image_path = os.path.join(model_image_folder, model_name + ".png")

    mask_folder = config.get("mask_folder", "../output/masks")
    mask_path = os.path.join(mask_folder, model_name + ".png")

    mask = imageio.imread(mask_path)
    # mask to 0 and 1 from 0 to 255
    if mask.max() > 1:
        mask = mask.astype(np.float32) / 255.0

    # save to temp debug
    save_img_to_temp(mask, config, "mask")

    image_size, target_height = mask.shape[1], mask.shape[0]
    logging.info("Mask size:", image_size, target_height)

    ### Initialize Renderers: ###############################################################################
    # Initialize a perspective camera.

    cameras, R, T, focal_px = calibrate_cameras(
        config=config, device=device, width=image_size, height=target_height
    )

    logging.debug("Camera calibrated with R: %s, T: %s, focal_px: %s", R, T, focal_px)

    # Initialize renderer ####################################################
    silhouette_renderer, phong_renderer, raster_settings = initialize_renderer(
        cameras=cameras,
        sigma=sigma,
        gamma=gamma,
        image_size=image_size,
        target_height=target_height,
        device=device,
    )

    pc_renderer = make_pointcloud_renderer(
        cameras=cameras,
        image_size=image_size,
        target_height=target_height,
        device=device,
    )

    ###########################################################################################################

    # Determine if object is on floor ####################################################
    # search if image with floor in name exists in folder, if so set obejct_on_floor accordingly
    object_on_floor = False

    floor_files = [f for f in os.listdir(mask_folder) if "floor" in f.lower()]


    if floor_files and config.get("use_floor_detection", True):
        floor_mask = imageio.imread(os.path.join(mask_folder, floor_files[0]))
        # mask to 0 and 1 from 0 to 255
        if floor_mask.max() > 1:
            floor_mask_binary = floor_mask.astype(np.float32) / 255.0


        object_on_floor = True
        try:
            # Get bounding box from object mask [x_min, y_min, x_max, y_max]
            obj_rows, obj_cols = np.where(mask > 0)
            if len(obj_rows) > 0:
                obj_bbox = [
                    obj_cols.min(),
                    obj_rows.min(),
                    obj_cols.max(),
                    obj_rows.max(),
                ]
            else:
                obj_bbox = [0, 0, 0, 0]

            # Get bounding box from floor mask [x_min, y_min, x_max, y_max]
            floor_rows, floor_cols = np.where(floor_mask_binary > 0)
            if len(floor_rows) > 0:
                floor_bbox = [
                    floor_cols.min(),
                    floor_rows.min(),
                    floor_cols.max(),
                    floor_rows.max(),
                ]
            else:
                floor_bbox = [0, 0, 0, 0]

            # Calculate IOU between bounding boxes
            iou = calculate_iou(obj_bbox, floor_bbox)

            # select names that should usually be on floor
            floor_object_names = config.get(
                "floor_object_names",
                ["chair", "sofa", "table", "couch", "bed", "cabinet", "desk", "sideboard", "dresser", "plant"],
            )

            if iou > 0 or any(name in model_name.lower() for name in floor_object_names):
                object_on_floor = True
                logging.info(
                    f"✅ '{model_name}' is ON FLOOR (bbox IOU={iou:.4f}) -> PlanarModel + plane alignment"
                )
                logging.debug(f"   Object bbox: {obj_bbox}, Floor bbox: {floor_bbox}")
            else:
                object_on_floor = False
                logging.debug(
                    f"❌ '{model_name}' NOT on floor (bbox IOU=0) -> RegularModel + non-planar init"
                )
                logging.debug(f"   Object bbox: {obj_bbox}, Floor bbox: {floor_bbox}")
        except Exception as e:
            logging.warning(f"Could not check floor overlap: {e}")
            object_on_floor = True
    else:
        logging.warning(f"Floor mask not found, or object isn't on floor.")

    ###############################################################################################################

    ###############################################

    # LOAD TARGET POINTCLOUD ####################################################

    point_cloud_folder = config.get("point_cloud_folder", "../output/pointclouds")
    pc_path = os.path.join(point_cloud_folder, model_name + ".ply")

    # load ply file
    target_pointcloud = trimesh.load(pc_path)
    # to numpy
    target_pointcloud = np.array(target_pointcloud.vertices, dtype=np.float32)
    target_pointcloud = np.ascontiguousarray(target_pointcloud, dtype=np.float64)

    # transform to torch tensor in shape N, 3
    target_pointcloud = torch.from_numpy(target_pointcloud).to(
        device=device, dtype=torch.float32
    )
    # import normals
    normals_folder = os.path.join(point_cloud_folder, "normals")
    normal_path = os.path.join(normals_folder, model_name + "_normals.ply")
    if os.path.exists(normal_path):
        target_normals = trimesh.load(normal_path)
        target_normals = np.array(target_normals.vertices, dtype=np.float32)
        target_normals = np.ascontiguousarray(target_normals, dtype=np.float64)

        # transform to torch tensor in shape N, 3
        target_normals = torch.from_numpy(target_normals).to(
            device=device, dtype=torch.float32
        )

    # temp render with pointcloud renderer for debug
    tp_PC = Pointclouds(points=[target_pointcloud], features=[torch.ones_like(target_pointcloud)])

    # estimate normals
    # tp_PC.to(device).estimate_normals(assign_to_self=True)

    print("Input pointcloud tensor shape:", target_pointcloud.shape)  # Should be (N, 3)
    print("tp_PC points_padded shape:", tp_PC.points_padded().shape)  # Should be (batch, N, 3)
    #print("tp_PC normals_padded shape:", tp_PC.normals_padded().shape)  # Should be (batch, N, 3)
    #print("First 5 normals:", tp_PC.normals_padded()[0, :5])

    pc_img = pc_renderer(point_clouds=tp_PC)
    pc_img_np = pc_img[0, ..., :3].cpu().numpy()
    save_img_to_temp(pc_img_np, config, "input_pointcloud_render")


    # ============================================================================
    # INITIALIZATION: Use non-planar OR planar based on floor IOU
    # ============================================================================
    # if not object_on_floor:
    # NOT on floor - use the old non-planar initialization code
    logging.debug("Using non-planar initialization (object not on floor)")

    # --- 1. Analyze both point cloud and mesh using the 2D-up OBB method ---
    pc_centroid, R_pc, extents_pc = get_oriented_bounding_box_2d_up(target_pointcloud)
    det = torch.det(R_pc)
    logging.debug(f"Determinant of PC OBB rotation matrix: {det}")

    if det < 0:
        # Reflection detected! Flip the sign of the first column (X axis)
        logging.debug("Mirroring detected in OBB rotation matrix. Correcting...")
        R_pc[:, 0] *= -1

    # For the mesh, we also analyze it while respecting its up-axis
    mesh_points_for_obb = sample_mesh_points(glb_object, num_samples=2048).squeeze(0)
    _, R_mesh, extents_mesh = get_oriented_bounding_box_2d_up(mesh_points_for_obb)

    # set R to identity for test
    if config.get("set_no_initial_rotation", True):
        R_pc = torch.eye(3, device=device)
        R_mesh = torch.eye(3, device=device)

    # --- 2. Align the mesh to the point cloud's OBB ---
    chair_verts = glb_object.verts_padded()  # (1, V, 3)

    # Center the mesh at the origin
    verts_centered = chair_verts - chair_verts.mean(dim=1, keepdim=True)

    # Undo the mesh's inherent ground-plane rotation
    verts_aligned = torch.matmul(verts_centered, R_mesh)

    # --- UNIFORM SCALING LOGIC ---
    volume_pc = extents_pc.prod()
    volume_mesh = extents_mesh.prod()
    scale_factor = (volume_pc / (volume_mesh + 1e-8)) ** (1 / 3)

    # Apply the single, uniform scale factor to the vertices
    verts_scaled = verts_aligned * scale_factor

    # Apply the target point cloud's ground-plane rotation
    verts_rotated = torch.matmul(verts_scaled, R_pc.T)

    # Apply the final translation to move the mesh to the target's centroid
    chair_verts_translated = verts_rotated + pc_centroid.unsqueeze(0).unsqueeze(1)
    logging.debug(
        f"Current chair verts translated shape: {chair_verts_translated.shape}"
    )

    # --- Use the Grid Search to Refine Rotation ---
    if config.get("use_rotation_grid_search", False):
        logging.debug("Applying rotation grid search for initial alignment...")
        initial_aligned_verts = chair_verts_translated.squeeze(0)

        # Find the best additional yaw rotation
        R_yaw_correction = find_best_initial_yaw(
            mesh_verts=initial_aligned_verts,
            target_points=target_pointcloud,
            num_angles=config.get("grid_rotation_steps", 36),
            use_chamfer=config.get("use_chamfer_distance", True),
            faces=glb_object.faces_list(),
            debug_save=(
                iteration == 12 and config.get("dump_rotation_search_debug", True)
            ),
            debug_dir=config.get("temp", "../output"),
            debug_prefix=f"{model_name}_yawgrid",
            centroid_method=config.get("rotation_center_method", "bbox"),
        )

        # Apply this final correction
        final_centroid = initial_aligned_verts.mean(dim=0, keepdim=True)
        verts_for_final_rotation = initial_aligned_verts - final_centroid
        chair_verts_translated = (
            torch.matmul(verts_for_final_rotation, R_yaw_correction.T) + final_centroid
        )
        # add batch dimension back
        chair_verts_translated = chair_verts_translated.unsqueeze(0)
        logging.debug(
            f"Chair verts after yaw correction shape: {chair_verts_translated.shape}"
        )

    # Rebuild the mesh
    glb_object = Meshes(
        verts=[chair_verts_translated[0]],
        faces=glb_object.faces_list(),
        textures=glb_object.textures,
    )
    # glb_object = clean_mesh(glb_object)

    # else:
    #     # ON floor - skip pre-3D init, will do plane alignment later
    #     logging.debug("Skipping pre-3D init (object on floor - will align to plane)")

    # Rebuild the mesh with original vertices (before plane alignment)
    # The plane-space alignment will happen next
    # glb_object = clean_mesh(glb_object)

    logging.debug("Mesh ready for plane-space alignment")

    ### DEBUG pointlcouds ############################################################################

    ####### DEBUG POINT CLOUDS ############################################################################

    visualize_pointclouds(
        target_pts=target_pointcloud,
        mesh=glb_object,
        num_samples=2048,
        pc_renderer=pc_renderer,
        config=config,
        device=device,
        camera_distance=1.0,
        image_size_x=image_size,
        image_size_y=target_height,
    )

    ######################################################################################

    # ============================================================================
    # PLANE EXTRACTION AND PROJECTION (only for objects ON floor)
    # ============================================================================
    if object_on_floor:
        # Extract floor plane from scene - only for objects on the floor
        logging.debug("=" * 80)
        logging.debug("EXTRACTING FLOOR PLANE FOR OBJECT PROJECTION")
        logging.debug("=" * 80)

        #logging.info("Shape binary floor mask:", floor_mask_binary.shape)
        #logging.info("Dtype binary floor mask:", floor_mask_binary.dtype)
        # floor_mask_binary is already a numpy array, just pass it directly
        plane_normal, plane_point, plane_to_world, world_to_plane, floor_pointcloud = (
            extract_and_fit_floor_plane(
                floor_mask=floor_mask_binary,  # Already np.ndarray with correct dtype
                vggt_cloud_path=config["vggt_cloud"],
                cameras=cameras,
                point_cloud_folder=point_cloud_folder,
                device=device,
            )
        )

        # Visualize the fitted plane overlaid on input image (only first time)
        if iteration == 0:
            visualize_plane_and_axes(
                plane_to_world_transform=plane_to_world,
                renderer=phong_renderer,
                cameras=cameras,
                output_path=os.path.join(
                    config.get("temp", "../output"), f"plane_debug_{model_name}.png"
                ),
                device=device,
                background_image_path=None,
                image_size=(image_size, target_height),
            )

        # ============================================================================
        # ALIGN OBJECT'S BOTTOM FACE TO PLANE USING BBOX + PCA
        # ============================================================================
        logging.debug("Aligning object's bottom face to plane using bbox + PCA...")

        # Get mesh vertices in world space
        chair_verts_world = glb_object.verts_padded()[0]  # (V, 3)

        # # Get target pointcloud (in world space)
        pc_centroid_world = target_pointcloud.mean(dim=0)

        # # Step 1: Scale mesh to match VGGT pointcloud volume
        # logging.debug("  Step 1: Scale to match VGGT pointcloud")
        # mesh_min_world = chair_verts_world.min(dim=0).values
        # mesh_max_world = chair_verts_world.max(dim=0).values
        # mesh_extents_world = mesh_max_world - mesh_min_world
        # mesh_volume = mesh_extents_world.prod()

        # pc_min_world = target_pointcloud.min(dim=0).values
        # pc_max_world = target_pointcloud.max(dim=0).values
        # pc_extents_world = pc_max_world - pc_min_world
        # pc_volume = pc_extents_world.prod()

        # scale_factor = (pc_volume / (mesh_volume + 1e-8)) ** (1/3)
        # logging.debug(f"    Scale factor: {scale_factor:.4f}")

        # # Apply scale from centroid
        # mesh_centroid_world = chair_verts_world.mean(dim=0)
        chair_verts_scaled = chair_verts_world  # (chair_verts_world - mesh_centroid_world) * scale_factor + mesh_centroid_world

        # Step 2: Get axis-aligned bounding box of scaled mesh (Y-up in world space)
        logging.debug("  Step 2: Extract bottom face of axis-aligned bbox")
        bbox_min = chair_verts_scaled.min(dim=0).values
        bbox_max = chair_verts_scaled.max(dim=0).values

        # Bottom face: 4 corners at Y_min
        bottom_corners_world = torch.tensor(
            [
                [
                    bbox_min[0],
                    bbox_min[1],
                    bbox_min[2],
                ],  # Corner 1: (x_min, y_min, z_min)
                [
                    bbox_max[0],
                    bbox_min[1],
                    bbox_min[2],
                ],  # Corner 2: (x_max, y_min, z_min)
                [
                    bbox_max[0],
                    bbox_min[1],
                    bbox_max[2],
                ],  # Corner 3: (x_max, y_min, z_max)
                [
                    bbox_min[0],
                    bbox_min[1],
                    bbox_max[2],
                ],  # Corner 4: (x_min, y_min, z_max)
            ],
            device=device,
        )

        bottom_centroid_world = bottom_corners_world.mean(dim=0)
        logging.debug(
            f"    Bottom face centroid: {bottom_centroid_world.cpu().numpy()}"
        )
        logging.debug(f"    Bottom Y-level: {bbox_min[1]:.4f}")

        # Step 3: Project bottom corners onto the fitted plane
        logging.debug("  Step 3: Project bottom corners onto fitted plane")
        # Plane equation: dot(point - plane_point, plane_normal) = 0
        # Project point P onto plane: P' = P - dot(P - plane_point, normal) * normal

        plane_point_world = plane_point.squeeze(0)  # (3,)
        plane_normal_world = plane_normal  # (3,)

        # Project each corner
        projected_corners = []
        for corner in bottom_corners_world:
            distance_to_plane = torch.dot(
                corner - plane_point_world, plane_normal_world
            )
            projected_corner = corner - distance_to_plane * plane_normal_world
            projected_corners.append(projected_corner)

        projected_corners = torch.stack(projected_corners)  # (4, 3)
        projected_centroid = projected_corners.mean(dim=0)

        logging.debug(f"    Projected centroid: {projected_centroid.cpu().numpy()}")

        # Step 4: PCA on projected corners to find orientation on the plane
        logging.debug("  Step 4: PCA on projected corners to find plane orientation")

        # Center the projected corners
        corners_centered = projected_corners - projected_centroid

        # Covariance matrix
        cov = (corners_centered.T @ corners_centered) / (corners_centered.shape[0] - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = eigenvalues.argsort(descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # The two largest eigenvectors define the plane orientation
        # PC1 = main direction, PC2 = perpendicular direction (both on plane)
        # plane_normal = PC3 (should match our fitted plane normal)

        pca_x = eigenvectors[:, 0]  # Main horizontal direction on plane
        pca_y = plane_normal_world  # Normal (vertical in plane coords)
        pca_z = torch.cross(pca_x, pca_y)  # Secondary horizontal direction
        pca_z = pca_z / torch.norm(pca_z)

        # Build rotation matrix: columns are the new basis vectors
        R_bbox_to_plane = torch.stack([pca_x, pca_y, pca_z], dim=1)  # (3, 3)

        if torch.det(R_bbox_to_plane) < 0:
            logging.debug("    Correcting handedness")
            R_bbox_to_plane[:, 0] *= -1

        logging.debug(f"    PCA eigenvalues: {eigenvalues.cpu().numpy()}")

        # Step 5: Tilt-only rotation to align object up with plane normal (preserve yaw)
        logging.debug(
            "  Step 5: Tilt object to align up with plane normal (preserve yaw)"
        )
        current_up = torch.tensor([0.0, 1.0, 0.0], device=device)
        target_up = plane_normal_world / (torch.norm(plane_normal_world) + 1e-9)

        # Axis for minimal rotation from current_up to target_up
        axis = torch.cross(current_up, target_up)
        sin_theta = torch.norm(axis)
        cos_theta = torch.clamp(torch.dot(current_up, target_up), -1.0, 1.0)

        if sin_theta < 1e-6:
            # Already aligned (or opposite). If opposite, rotate 180 around any axis perpendicular to up.
            if cos_theta < 0:
                # Choose X-axis for 180deg flip
                axis_n = torch.tensor([1.0, 0.0, 0.0], device=device)
                K = torch.tensor(
                    [
                        [0, -axis_n[2], axis_n[1]],
                        [axis_n[2], 0, -axis_n[0]],
                        [-axis_n[1], axis_n[0], 0],
                    ],
                    device=device,
                    dtype=chair_verts_scaled.dtype,
                )
                R_tilt = -torch.eye(3, device=device, dtype=chair_verts_scaled.dtype)
                R_tilt[1, 1] = 1.0  # 180 around X flips Y,Z; preserve handedness
            else:
                R_tilt = torch.eye(3, device=device, dtype=chair_verts_scaled.dtype)
        else:
            axis_n = axis / sin_theta
            # Rodrigues' rotation formula components
            K = torch.tensor(
                [
                    [0, -axis_n[2], axis_n[1]],
                    [axis_n[2], 0, -axis_n[0]],
                    [-axis_n[1], axis_n[0], 0],
                ],
                device=device,
                dtype=chair_verts_scaled.dtype,
            )
            theta = torch.atan2(sin_theta, cos_theta)
            I = torch.eye(3, device=device, dtype=chair_verts_scaled.dtype)
            R_tilt = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)

        # Apply tilt around bottom centroid
        verts_centered = chair_verts_scaled - bottom_centroid_world
        verts_rotated = torch.matmul(verts_centered, R_tilt.T) + bottom_centroid_world

        # Debug: check alignment quality and yaw preservation intent
        new_up = torch.matmul(current_up.unsqueeze(0), R_tilt.T).squeeze(0)
        align_cos = torch.dot(new_up / (torch.norm(new_up) + 1e-9), target_up)
        logging.debug(
            f"    Up alignment cos(angle): {align_cos.item():.6f} (1.0 means perfectly aligned)"
        )
        # Yaw preservation: rotation axis has no component along target_up ideally
        yaw_component = (
            torch.abs(torch.dot(axis / (sin_theta + 1e-9), target_up))
            if sin_theta >= 1e-6
            else 0.0
        )
        logging.debug(
            f"    Yaw-change component around plane normal: {float(yaw_component):.6f} (near 0 preserves yaw)"
        )

        # Step 6: Position at target centroid
        logging.debug("  Step 6: Position at target location")

        # Translate to target position
        translation = pc_centroid_world - bottom_centroid_world
        verts_positioned = verts_rotated + translation

        # Step 7: Place bottom on the plane
        logging.debug("  Step 7: Place bottom on fitted plane")

        # Find the lowest point after rotation/positioning
        lowest_idx = verts_positioned[:, 1].argmin()
        lowest_point = verts_positioned[lowest_idx]

        # Calculate distance from lowest point to plane
        distance_to_plane = torch.dot(
            lowest_point - plane_point_world, plane_normal_world
        )

        # Move entire mesh along plane normal to place bottom on plane
        verts_on_plane = verts_positioned - distance_to_plane * plane_normal_world

        # Verify
        new_lowest = verts_on_plane[verts_on_plane[:, 1].argmin()]
        final_distance = torch.dot(new_lowest - plane_point_world, plane_normal_world)

        logging.debug(f"    Distance to plane before: {distance_to_plane:.4f}m")
        logging.debug(
            f"    Distance to plane after: {final_distance:.6f}m (should be ~0)"
        )
        logging.debug(f"    Bottom point: {new_lowest.cpu().numpy()}")

        # Rebuild mesh
        glb_object = Meshes(
            verts=[verts_on_plane],
            faces=glb_object.faces_list(),
            textures=glb_object.textures,
        )
        glb_object = clean_mesh(glb_object)

        logging.debug("✅ Object bottom face aligned to plane using bbox + PCA")
    else:
        # Object NOT on floor - mesh already initialized with non-planar code above
        logging.debug("Skipping plane alignment (object not on floor)")
        plane_to_world = torch.eye(4, device=device)  # Dummy for RegularModel

    # ============================================================================
    # MODEL SELECTION: PlanarModel for floor objects, RegularModel otherwise
    # ============================================================================

    # TODO : add bbox loss against background model
    # create mask that is whole image in as ndarray
    background_path = os.path.join(
        config.get("output_vggt", "../output/vggt/sparse"), "points_emptyRoom.ply"
    )

    def save_bbox_as_obj(bbox, out_path):
        # bbox: [min_x, min_y, min_z, max_x, max_y, max_z]
        min_xyz = np.array(bbox[:3])
        max_xyz = np.array(bbox[3:])
        # 8 corners
        corners = np.array(
            [
                [min_xyz[0], min_xyz[1], min_xyz[2]],
                [max_xyz[0], min_xyz[1], min_xyz[2]],
                [max_xyz[0], max_xyz[1], min_xyz[2]],
                [min_xyz[0], max_xyz[1], min_xyz[2]],
                [min_xyz[0], min_xyz[1], max_xyz[2]],
                [max_xyz[0], min_xyz[1], max_xyz[2]],
                [max_xyz[0], max_xyz[1], max_xyz[2]],
                [min_xyz[0], max_xyz[1], max_xyz[2]],
            ]
        )
        # 12 faces (each as 2 triangles)
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # bottom
                [4, 5, 6],
                [4, 6, 7],  # top
                [0, 1, 5],
                [0, 5, 4],  # front
                [2, 3, 7],
                [2, 7, 6],  # back
                [1, 2, 6],
                [1, 6, 5],  # right
                [0, 3, 7],
                [0, 7, 4],  # left
            ]
        )
        mesh = trimesh.Trimesh(vertices=corners, faces=faces)
        mesh.export(out_path)

    if os.path.exists(background_path):
        vggt_cloud_bg = background_path
        from scene_optimization.mesh_pointclouds import set_vggt_cloud

        background_cloud = set_vggt_cloud(
            background_path,
            vggt_scene_scale=config.get("vggt_scene_scale", 2.0),
            device=device,
        )

        # save to temp debug
        save_point_cloud(
            background_cloud, "../tmp/background_cloud.ply", blender_readable=False
        )

        # tensor to numpy
        background_cloud = background_cloud.cpu().numpy().astype(np.float64)
        min_xyz = background_cloud.min(axis=0)
        max_xyz = background_cloud.max(axis=0)
        bg_bbox_extents = config.get("background_bbox_extents", -0.025)
        background_bbox = [
            min_xyz[0] - bg_bbox_extents,
            min_xyz[1] - bg_bbox_extents,
            min_xyz[2] - bg_bbox_extents,
            max_xyz[0] + bg_bbox_extents,
            max_xyz[1] + bg_bbox_extents,
            max_xyz[2] + bg_bbox_extents,
        ]
        save_bbox_as_obj(background_bbox, "../tmp/background_bbox.obj")


    # add gaussian blur to mask
    from scipy.ndimage import gaussian_filter
    mask = gaussian_filter(mask, sigma=1)
    

    # Initialize model based on object type
    if object_on_floor:
        model = PlanarModel(
            meshes=glb_object,
            renderer=silhouette_renderer,
            plane_to_world_transform=plane_to_world,
            cameras=cameras,
            mask=mask,
            target_pointcloud=tp_PC, #target_pointcloud,
            config=config,
            device=device,
            background_bbox=background_bbox if "background_bbox" in locals() else None,
        ).to(device)
    else:
        model = RegularModel(
            meshes=glb_object,
            renderer=silhouette_renderer,
            cameras=cameras,
            mask=mask,
            target_pointcloud=target_pointcloud,
            config=config,
            device=device,
            background_bbox=background_bbox if "background_bbox" in locals() else None,
        ).to(device)

    # Create optimizer with translation, full rotation, and scale parameters
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config.get("learning_rate", 0.01),
    )

    frames = []  # Store frames for GIF
    # Load background image for GIF composition
    background = imageio.imread(model_image_path)  # Load the actual image
    background = resize(background, (image_size, target_height)).astype(np.float32)
    # Normalize to [0, 1] if needed
    if background.max() > 1:
        background = background.astype(np.float32) / 255.0

    # Early stopping configuration
    grad_threshold = config.get("early_stop_grad_threshold", 5e-3)
    min_iterations = config.get("early_stop_min_iterations", 100)

    # Optimization loop ##################################################################################################
    # debug:
    for i in tqdm(range(config.get("max_iterations", 100))):
        optimizer.zero_grad()
        loss, updated_model, P = model()

        # Visualize silhouette and object+plane periodically
        if i % 10 == 0:
            save_img_to_temp(P.detach().cpu(), config, f"current_silhouette")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # TODO : Care


        # Calculate total gradient magnitude for early stopping
        total_grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm**0.5

        # # DEBUG LOGGING
        logging.debug("\n=== GRADIENT DIAGNOSTICS ===")
        logging.debug("Total gradient norm: %f", total_grad_norm)

        if not object_on_floor:
            logging.debug(
                "Rotation grad norm: %f",
                (
                    model.rotation.grad.norm().item()
                    if model.rotation.grad is not None
                    else 0
                ),
            )
            logging.debug(
                "Translation grad norm: %f",
                (
                    model.translation.grad.norm().item()
                    if model.translation.grad is not None
                    else 0
                ),
            )
            logging.debug(
                "Scale grad norm: %f",
                model.scale.grad.norm().item() if model.scale.grad is not None else 0,
            )

            # Check if gradients are NaN
            logging.debug(
                "Grads contain NaN: %s",
                any(
                    torch.isnan(p.grad).any()
                    for p in model.parameters()
                    if p.grad is not None
                ),
            )

            # Visualize gradient direction
            if model.translation.grad is not None:
                logging.debug(
                    "Translation grad direction: %s",
                    model.translation.grad.detach().cpu().numpy(),
                )

        optimizer.step()

        # Early stopping check
        if i >= min_iterations and total_grad_norm < grad_threshold:
            logging.debug(
                f"Early stopping at iteration {i}: gradient norm {total_grad_norm:.2e} below threshold {grad_threshold:.2e}"
            )
            break

        # save every 10th iteration to the gif
        frames = add_frames_gif(
            frames=frames,
            step=i,
            step_size=5,
            phong_renderer=phong_renderer,
            model=updated_model,
            background=None,  # background
        )
    #####################################################################################################################

    logging.debug(
        f"Optimization completed at iteration {i+1}/{config.get('max_iterations', 100)}"
    )

    logging.debug(
        f"Optimization complete for model : {model_name} ###########################."
    )
    if not object_on_floor:
        logging.debug("Final translation:", model.translation.detach().cpu().numpy())
        logging.debug("Final rotation:", model.rotation.detach().cpu().numpy())
        logging.debug("Final scale:", model.scale.detach().cpu().numpy())

    # Try saving final mesh as glb with correct naming
    save_glb_mesh(
        glb_dir=final_glb_path, model_name=model_name, glb_object=updated_model
    )

    # Save the final rendered image as a gif
    gif_filename = os.path.join(gif_folder, f"{model_name}.gif")
    imageio.mimsave(gif_filename, frames, duration=0.3, loop=0)
