import logging
import torch
import trimesh
import numpy as np
import logging

from utils.global_utils import (
    B2P
)

def get_model_vggt_cloud(mask: np.ndarray, vggt_cloud_path: str, cameras, device="cpu"):
    """
    Given a binary mask (H, W) in image coordinates, return the corresponding
    cropped point cloud in world coordinates.
    """
    # Load VGGT cloud (PLY)
    pc_trimesh = trimesh.load(vggt_cloud_path)
    points3d_raw = torch.from_numpy(pc_trimesh.vertices).float().to(device)  # (N, 3)
    
    # ✅ DIAGNOSTIC: Save raw VGGT data to check coordinate system
    logging.debug(f"[DEBUG] Raw VGGT before any transform:")
    logging.debug(f"  Y range: [{points3d_raw[:, 1].min():.4f}, {points3d_raw[:, 1].max():.4f}]")
    logging.debug(f"  Y std: {points3d_raw[:, 1].std():.4f}")

    R_np, t_np = B2P(np.eye(4))
    R_torch = torch.from_numpy(R_np).float().to(device)
    t_torch = torch.from_numpy(t_np).float().to(device)
    # Apply to the raw point cloud
    points3d = points3d_raw @ R_torch.T + t_torch

    logging.debug(f"[DEBUG] After B2P transform:")
    logging.debug(f"  Y range: [{points3d[:, 1].min():.4f}, {points3d[:, 1].max():.4f}]")
    logging.debug(f"  Y std: {points3d[:, 1].std():.4f}")

    points3d[:, 1] *= -1

    logging.debug(f"[DEBUG] After Y-flip:")
    logging.debug(f"  Y range: [{points3d[:, 1].min():.4f}, {points3d[:, 1].max():.4f}]")
    logging.debug(f"  Y std: {points3d[:, 1].std():.4f}")

    # if mask is none give back full pointcloud
    if mask is None:
        return points3d

    # Project to image space
    proj = cameras.transform_points_screen(points3d[None], image_size=mask.shape[:2])
    proj = proj[0]  # (N, 3)
    x, y = proj[:, 0], proj[:, 1]

    # Convert to integer indices
    x_int = x.round().long()
    y_int = y.round().long()


    
    H, W = mask.shape[:2]

    # ✅ Keep only points that fall inside image bounds
    valid = (x_int >= 0) & (x_int < W) & (y_int >= 0) & (y_int < H)
    x_int = x_int[valid]
    y_int = y_int[valid]
    points3d = points3d[valid]

    # ✅ Apply mask filtering
    mask_torch = torch.from_numpy(mask).to(device)
    valid_mask = mask_torch[y_int, x_int] > 0
    points3d = points3d[valid_mask]

    logging.debug(
       f"Filtered point cloud: {points3d.shape} "
       f"with min/max bounds {points3d.min(dim=0).values} / {points3d.max(dim=0).values}"
    )

    # Return filtered point cloud in world space
    return points3d



def filter_points_by_quantile(points: torch.Tensor, q: float = 0.02) -> torch.Tensor:
    """
    Filters a point cloud by removing outlier points based on axis-aligned quantiles.

    Args:
        points (torch.Tensor): The input point cloud of shape (N, 3).
        q (float): The quantile to trim from both ends of each axis (e.g., 0.02 means trim 2% from the min and 2% from the max).

    Returns:
        torch.Tensor: The filtered, more robust point cloud.
    """
    if points.numel() == 0:
        return points

    # Calculate the lower and upper quantile bounds for each axis (X, Y, Z)
    lower_bounds = torch.quantile(points, q, dim=0)
    upper_bounds = torch.quantile(points, 1 - q, dim=0)

    # Create a boolean mask for points that fall within the bounds for ALL three axes
    mask = (points >= lower_bounds) & (points <= upper_bounds)
    mask = mask.all(dim=1) # Point must be inside bounds on X AND Y AND Z

    # Apply the mask to get the filtered point cloud
    filtered_points = points[mask]
    
    # It's possible to filter out everything if q is too high, so we add a fallback
    if filtered_points.numel() == 0:
        return points # Return original if the filtering was too aggressive

    return filtered_points



def filter_dbscan(target_pointcloud: torch.Tensor, eps: float = 0.05, min_samples: int = 10) -> torch.Tensor:
    """
    Applies DBSCAN clustering to filter the point cloud.

    Args:
        target_pointcloud (torch.Tensor): The input point cloud of shape (N, 3).
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        torch.Tensor: The filtered point cloud.
    """
    from sklearn.cluster import DBSCAN

    # Assuming 'target_pointcloud' is your (N, 3) torch.Tensor on the GPU
    points_np = target_pointcloud.cpu().numpy()

    # DBSCAN parameters are crucial.
    # eps: The max distance between two samples for one to be considered as in the neighborhood of the other.
    # A good starting point for 'eps' is the average distance of every point to its 5th or 10th nearest neighbor.
    # min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points_np)

    # The 'labels' array tells you which cluster each point belongs to. Noise is labeled -1.
    labels = db.labels_

    # Find the largest cluster (ignoring the noise label -1)
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(counts) > 0:
        largest_cluster_label = unique_labels[counts.argmax()]
        
        # Create a mask for the points in the largest cluster
        mask = (labels == largest_cluster_label)
        
        # Your cleaned point cloud is the largest cluster
        cleaned_target_pointcloud = target_pointcloud[torch.from_numpy(mask).to(target_pointcloud.device)]
    else:
        # Fallback if no clusters were found
        cleaned_target_pointcloud = target_pointcloud

    # Now use 'cleaned_target_pointcloud' for your scale and centroid calculations.
    return cleaned_target_pointcloud
