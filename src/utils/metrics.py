import torch
from pytorch3d.ops import knn_points

# from chamfer_distance import ChamferDistance

# Code from MIDI https://github.com/VAST-AI-Research/MIDI-3D


# # Compute metrics
# def compute_chamfer_distance(
#     pred: torch.Tensor,
#     gt: torch.Tensor,
#     chamfer_distance_cls: torch.nn.Module = None,
#     device: torch.device = torch.device("cuda:0"),
# ):
#     """
#     Compute Chamfer Distance between predicted and ground truth point clouds.

#     Args:
#         pred (torch.Tensor): Predicted point cloud of shape (B, N, 3)
#         gt (torch.Tensor): Ground truth point cloud of shape (B, M, 3)

#     Returns:
#         torch.Tensor: Chamfer Distance for each batch
#     """
#     if chamfer_distance_cls is None:
#         chamfer_distance_cls = ChamferDistance().to(device)

#     # Compute Chamfer Distance
#     dist1, dist2 = chamfer_distance_cls(pred, gt)

#     # Average the distances and ensure they are on the correct device
#     dist1 = dist1.mean(dim=1).to(device)
#     dist2 = dist2.mean(dim=1).to(device)

#     return dist1 + dist2, dist1, dist2


def compute_fscore(pred, gt, tau=0.1, chunk_size=2048):
    """
    Compute F-Score between predicted and ground truth point clouds.

    Args:
        pred (torch.Tensor): Predicted point cloud of shape (B, N, 3)
        gt (torch.Tensor): Ground truth point cloud of shape (B, M, 3)
        tau (float): Distance threshold for F-score
        chunk_size (int): Size of chunks to process at a time

    Returns:
        torch.Tensor: F-Score for each batch
    """
    B, N, _ = pred.shape
    _, M, _ = gt.shape

    # Initialize tensors to store minimal distances
    min_dists_pred_to_gt = torch.zeros(B, N, device=pred.device)
    min_dists_gt_to_pred = torch.zeros(B, M, device=gt.device)

    for b in range(B):
        pred_b = pred[b]  # (N, 3)
        gt_b = gt[b]  # (M, 3)

        # Process pred to gt in chunks
        for i in range(0, N, chunk_size):
            pred_chunk = pred_b[i : i + chunk_size]  # (chunk_size, 3)
            # Compute distances and get minimum
            dists = torch.cdist(
                pred_chunk.unsqueeze(0), gt_b.unsqueeze(0), p=2
            )  # (1, chunk_size, M)
            min_dists = dists.min(dim=2).values.squeeze(0)  # (chunk_size,)
            min_dists_pred_to_gt[b, i : i + chunk_size] = min_dists

        # Process gt to pred in chunks
        for i in range(0, M, chunk_size):
            gt_chunk = gt_b[i : i + chunk_size]  # (chunk_size, 3)
            # Compute distances and get minimum
            dists = torch.cdist(
                gt_chunk.unsqueeze(0), pred_b.unsqueeze(0), p=2
            )  # (1, chunk_size, N)
            min_dists = dists.min(dim=2).values.squeeze(0)  # (chunk_size,)
            min_dists_gt_to_pred[b, i : i + chunk_size] = min_dists

    # Determine matches within tau distance
    precision_matches = (min_dists_pred_to_gt < tau).float()  # (B, N)
    recall_matches = (min_dists_gt_to_pred < tau).float()  # (B, M)

    # Calculate precision and recall
    precision = precision_matches.sum(dim=1) / N  # (B,)
    recall = recall_matches.sum(dim=1) / M  # (B,)

    # Calculate F-Score
    fscore = (
        2 * (precision * recall) / (precision + recall + 1e-8)
    )  # Avoid division by zero

    return fscore


def voxelize(points, voxel_size, grid_size, min_bound):
    """
    Voxelize the point cloud.

    Args:
        points (torch.Tensor): Point cloud of shape (B, N, 3)
        voxel_size (float): Size of each voxel
        grid_size (int): Number of voxels along each dimension
        min_bound (torch.Tensor): Minimum bounds of the grid

    Returns:
        torch.Tensor: Voxel grid of shape (B, grid_size, grid_size, grid_size) with boolean type
    """
    # Shift points to positive grid and scale
    scaled_points = (points - min_bound) / voxel_size
    indices = torch.floor(scaled_points).long()

    # Clamp indices to grid size
    indices = indices.clamp(0, grid_size - 1)

    # Create voxel grid
    voxel_grid = torch.zeros(
        (points.shape[0], grid_size, grid_size, grid_size),
        device=points.device,
        dtype=torch.bool,
    )
    for b in range(points.shape[0]):
        voxel_grid[b, indices[b, :, 0], indices[b, :, 1], indices[b, :, 2]] = True

    return voxel_grid


def compute_volume_iou(pred, gt, voxel_size=0.05, grid_size=64, mode="pcd"):
    """
    Compute Volume IoU between predicted and ground truth point clouds.

    Args:
        pred (torch.Tensor): Predicted point cloud of shape (B, N, 3)
        gt (torch.Tensor): Ground truth point cloud of shape (B, N, 3)
        voxel_size (float): Size of each voxel
        grid_size (int): Number of voxels along each dimension
        mode (str): Mode of computing volume iou, either "pcd" or "bbox"

    Returns:
        torch.Tensor: Volume IoU for each batch
    """
    if mode == "pcd":
        # Define the grid bounds
        min_bound = torch.min(torch.min(pred, dim=1).values, dim=0).values
        max_bound = torch.max(torch.max(pred, dim=1).values, dim=0).values
        min_bound = torch.min(min_bound, torch.min(gt, dim=1).values.min(dim=0).values)
        max_bound = torch.max(max_bound, torch.max(gt, dim=1).values.max(dim=0).values)

        # Voxelize the point clouds
        pred_voxels = voxelize(pred, voxel_size, grid_size, min_bound)
        gt_voxels = voxelize(gt, voxel_size, grid_size, min_bound)

        # Compute intersection and union
        intersection = (pred_voxels & gt_voxels).sum(dim=(1, 2, 3)).float()
        union = (pred_voxels | gt_voxels).sum(dim=(1, 2, 3)).float()

        # Compute IoU
        iou = intersection / (union + 1e-8)

    elif mode == "bbox":
        # Compute bounding boxes
        pred_min = pred.min(dim=1).values
        pred_max = pred.max(dim=1).values
        gt_min = gt.min(dim=1).values
        gt_max = gt.max(dim=1).values

        # Compute intersection
        intersection_min = torch.max(pred_min, gt_min)
        intersection_max = torch.min(pred_max, gt_max)
        inter_dims = (intersection_max - intersection_min).clamp(min=0)
        inter_vol = inter_dims[:, 0] * inter_dims[:, 1] * inter_dims[:, 2]

        # Compute union
        pred_dims = (pred_max - pred_min).clamp(min=0)
        pred_vol = pred_dims[:, 0] * pred_dims[:, 1] * pred_dims[:, 2]  # (B,)
        gt_dims = (gt_max - gt_min).clamp(min=0)
        gt_vol = gt_dims[:, 0] * gt_dims[:, 1] * gt_dims[:, 2]

        # Compute IoU
        union_vol = pred_vol + gt_vol - inter_vol
        iou = inter_vol / (union_vol + 1e-8)

    else:
        raise ValueError(f"Invalid mode: {mode}")

    return iou


# Utils
def compute_nearest_neighbors(src, dst):
    """
    Compute the nearest neighbors from src to dst using PyTorch3D's knn_points.

    Args:
        src (torch.Tensor): Source point cloud of shape (B, N, 3)
        dst (torch.Tensor): Destination point cloud of shape (B, M, 3)

    Returns:
        torch.Tensor: Nearest neighbor indices in dst for each point in src, shape (B, N)
    """
    # Compute nearest neighbors with k=1
    knn = knn_points(src, dst, K=1)
    nn_indices = knn.idx.squeeze(-1)  # Shape (B, N)
    return nn_indices


def compute_rigid_transform(A, B):
    """
    Compute the rigid transformation (R, t) that aligns A to B.

    Args:
        A (torch.Tensor): Source point cloud of shape (B, N, 3)
        B (torch.Tensor): Target point cloud of shape (B, N, 3)

    Returns:
        R (torch.Tensor): Rotation matrices of shape (B, 3, 3)
        t (torch.Tensor): Translation vectors of shape (B, 3, 1)
    """
    # Compute centroids
    centroid_A = A.mean(dim=1, keepdim=True)
    centroid_B = B.mean(dim=1, keepdim=True)

    # Center the points
    AA = A - centroid_A
    BB = B - centroid_B

    # Compute covariance matrix
    H = torch.matmul(AA.transpose(1, 2), BB)

    # Compute SVD
    U, S, Vt = torch.svd(H)
    V = Vt.transpose(1, 2)
    R = torch.matmul(V, U.transpose(1, 2))

    # Handle reflection case
    det_R = torch.det(R)
    ones = torch.ones_like(det_R)
    eye = torch.eye(3, device=R.device).unsqueeze(0).repeat(R.size(0), 1, 1)
    diag = torch.diag_embed(torch.stack([ones, ones, det_R], dim=1))
    R = torch.matmul(torch.matmul(V, diag), U.transpose(1, 2))

    # Compute translation
    t = centroid_B.transpose(1, 2) - torch.matmul(
        R, centroid_A.transpose(1, 2)
    )  # (B, 3, 1)

    return R, t


def icp(src, dst, max_iterations=20, tolerance=1e-5):
    """
    Perform ICP algorithm to align src to dst.

    Args:
        src (torch.Tensor): Source point cloud of shape (B, N, 3)
        dst (torch.Tensor): Destination point cloud of shape (B, M, 3)
        max_iterations (int): Maximum number of ICP iterations
        tolerance (float): Convergence tolerance

    Returns:
        torch.Tensor: Transformed source point cloud
        torch.Tensor: Cumulative rotation matrices of shape (B, 3, 3)
        torch.Tensor: Cumulative translation vectors of shape (B, 3, 1)
    """
    src_transformed = src.clone()
    prev_error = float("inf")

    # Initialize cumulative rotation and translation
    cumulative_R = (
        torch.eye(3, device=src.device).unsqueeze(0).repeat(src.shape[0], 1, 1)
    )
    cumulative_t = torch.zeros((src.shape[0], 3, 1), device=src.device)

    for i in range(max_iterations):
        # Step 1: Find nearest neighbors
        nn_indices = compute_nearest_neighbors(src_transformed, dst)
        B, N, _ = src_transformed.shape

        # Gather nearest neighbors : Orig
        # nearest_neighbors = torch.gather(
        #     dst, 1, nn_indices.unsqueeze(-1).expand(-1, -1, 3)
        # )

        B, N, _ = src_transformed.shape
        batch_indices = torch.arange(B, device=src.device).unsqueeze(1).expand(-1, N)
        nearest_neighbors = dst[batch_indices, nn_indices]

        # Step 2: Compute the transformation
        R, t = compute_rigid_transform(src_transformed, nearest_neighbors)

        # Step 3: Apply transformation
        src_transformed = torch.matmul(
            src_transformed, R.transpose(1, 2)
        ) + t.transpose(1, 2)

        # Update cumulative transformation
        cumulative_R = torch.matmul(R, cumulative_R)
        cumulative_t = torch.matmul(R, cumulative_t) + t

        # Step 4: Check for convergence
        mean_error = torch.mean(torch.norm(src_transformed - nearest_neighbors, dim=2))
        if torch.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return src_transformed, cumulative_R, cumulative_t


# orig
# def icp(src, dst, max_iterations=20, tolerance=1e-5):
#     """
#     Perform ICP algorithm to align src to dst.

#     Args:
#         src (torch.Tensor): Source point cloud of shape (B, N, 3)
#         dst (torch.Tensor): Destination point cloud of shape (B, M, 3)
#         max_iterations (int): Maximum number of ICP iterations
#         tolerance (float): Convergence tolerance

#     Returns:
#         torch.Tensor: Transformed source point cloud
#         torch.Tensor: Cumulative rotation matrices of shape (B, 3, 3)
#         torch.Tensor: Cumulative translation vectors of shape (B, 3, 1)
#     """
#     src_transformed = src.clone()
#     prev_error = float("inf")

#     # Initialize cumulative rotation and translation
#     cumulative_R = (
#         torch.eye(3, device=src.device).unsqueeze(0).repeat(src.shape[0], 1, 1)
#     )
#     cumulative_t = torch.zeros((src.shape[0], 3, 1), device=src.device)

#     for i in range(max_iterations):
#         # Step 1: Find nearest neighbors
#         nn_indices = compute_nearest_neighbors(src_transformed, dst)
#         B, N, _ = src_transformed.shape

#         # Gather nearest neighbors
#         nearest_neighbors = torch.gather(
#             dst, 1, nn_indices.unsqueeze(-1).expand(-1, -1, 3)
#         )

#         # Step 2: Compute the transformation
#         R, t = compute_rigid_transform(src_transformed, nearest_neighbors)

#         # Step 3: Apply transformation
#         src_transformed = torch.matmul(
#             src_transformed, R.transpose(1, 2)
#         ) + t.transpose(1, 2)

#         # Update cumulative transformation
#         cumulative_R = torch.matmul(R, cumulative_R)
#         cumulative_t = torch.matmul(R, cumulative_t) + t

#         # Step 4: Check for convergence
#         mean_error = torch.mean(torch.norm(src_transformed - nearest_neighbors, dim=2))
#         if torch.abs(prev_error - mean_error) < tolerance:
#             break
#         prev_error = mean_error

#     return src_transformed, cumulative_R, cumulative_t
