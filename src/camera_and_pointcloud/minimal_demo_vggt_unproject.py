import random
import sys
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F
import yaml
import shutil
# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from pathlib import Path
import trimesh
import pycolmap
import argparse

from pycolmap import Rigid3d

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track

from utils.global_utils import load_config, save_point_cloud, B2P
import logging

# save out depth map
import imageio

#### Added ##################################################################

def align_pointclouds_obb(source_points, target_points):
    """
    Align source point cloud to target point cloud using oriented bounding boxes.
    This computes a transformation (rotation + translation + scale) that aligns
    the source OBB to match the target OBB.
    
    This is much more robust than PCA alignment when point clouds differ significantly
    (e.g., one has objects removed), as it only matches the overall shape/extent.
    
    Parameters
    ----------
    source_points : np.ndarray
        Source point cloud to be aligned (N, 3)
    target_points : np.ndarray
        Target point cloud (M, 3) - the reference
        
    Returns
    -------
    aligned_points : np.ndarray
        Aligned source points (N, 3)
    scale : float
        Scale factor applied
    R : np.ndarray
        Rotation matrix (3, 3) - identity matrix (no rotation)
    t : np.ndarray
        Translation vector (3,)
    """
    print(f"Aligning point clouds using scale + translation only - Source: {source_points.shape}, Target: {target_points.shape}")
    
    # 1. Compute centers
    source_center = np.mean(source_points, axis=0)
    target_center = np.mean(target_points, axis=0)
    
    source_centered = source_points - source_center
    target_centered = target_points - target_center
    
    # 2. Compute axis-aligned extents (no rotation needed)
    source_min = np.min(source_centered, axis=0)
    source_max = np.max(source_centered, axis=0)
    target_min = np.min(target_centered, axis=0)
    target_max = np.max(target_centered, axis=0)
    
    source_extents = source_max - source_min
    target_extents = target_max - target_min
    
    # 3. Compute scale factor (average scale across all axes OR per-axis)
    # Avoid division by zero
    scale_factors = np.divide(target_extents, source_extents, 
                              out=np.ones_like(target_extents), 
                              where=source_extents > 1e-6)
    
    # Option 1: Use average scale (uniform scaling - preserves shape)
    scale = np.mean(scale_factors)
    
    # Option 2: Use per-axis scale (non-uniform scaling - matches bounds exactly)
    # Uncomment the next line to use per-axis scaling instead:
    scale = scale_factors  # This will be a (3,) array
    
    print(f"  Scale factors per axis: {scale_factors}")
    # if isinstance(scale, np.ndarray):
    #     print(f"  Using per-axis scale: {scale}")
    # else:
    #     print(f"  Using average scale: {scale:.4f}")
    print(f"  Source extents: {source_extents}")
    print(f"  Target extents: {target_extents}")
    
    # 4. Apply transformation: scale -> translate (NO rotation)
    aligned_centered = source_centered * scale  # Broadcasting works for both uniform and per-axis
    aligned_points = aligned_centered + target_center
    
    # No rotation matrix (identity)
    R = np.eye(3)
    
    # Translation vector
    t = target_center - (source_center * scale)
    
    print(f"Scale + Translation alignment complete")
    print(f"  Source center: {source_center}, Target center: {target_center}")
    print(f"  Translation: {t}")
    
    return aligned_points, scale, R, t


def align_pointclouds_pca(source_points, target_points):
    """
    Align source point cloud to target point cloud using PCA.
    This computes a transformation (rotation + translation) that aligns
    the principal axes of the source to match the target.
    
    NOTE: This may fail when point clouds differ significantly (e.g., objects removed).
    Consider using align_pointclouds_obb() instead for more robustness.
    
    Parameters
    ----------
    source_points : np.ndarray
        Source point cloud to be aligned (N, 3)
    target_points : np.ndarray
        Target point cloud (M, 3) - the reference
        
    Returns
    -------
    aligned_points : np.ndarray
        Aligned source points (N, 3)
    R : np.ndarray
        Rotation matrix (3, 3)
    t : np.ndarray
        Translation vector (3,)
    """
    from sklearn.decomposition import PCA
    
    print(f"Aligning point clouds using PCA - Source: {source_points.shape}, Target: {target_points.shape}")
    
    # Center both point clouds
    source_center = np.mean(source_points, axis=0)
    target_center = np.mean(target_points, axis=0)
    
    source_centered = source_points - source_center
    target_centered = target_points - target_center
    
    # Compute PCA for both point clouds
    pca_source = PCA(n_components=3)
    pca_target = PCA(n_components=3)
    
    pca_source.fit(source_centered)
    pca_target.fit(target_centered)
    
    # Get principal axes (components are rows in sklearn PCA)
    axes_source = pca_source.components_  # (3, 3)
    axes_target = pca_target.components_  # (3, 3)
    
    # Compute rotation from source axes to target axes
    # R = target_axes @ source_axes.T
    R = axes_target.T @ axes_source
    
    # Apply rotation to centered source points
    aligned_centered = source_centered @ R.T
    
    # Translate to target center
    aligned_points = aligned_centered + target_center
    
    # Translation vector
    t = target_center - (source_center @ R.T)
    
    print(f"PCA alignment complete - Rotation determinant: {np.linalg.det(R):.4f}")
    print(f"Source center: {source_center}, Target center: {target_center}")
    
    return aligned_points, R, t


#############################################################################
# print current working dir
def export_vggt_data(
    config: dict, 

) -> None:
    """
    Parameters
    ----------
    config: dict
        Configuration dictionary containing paths and parameters.
    device: torch.device
        The device to run the export on (CPU or GPU).
    """
    # Load the reconstruction

    reconstruction_path = config.get("output_vggt", "../output/vggt/sparse")
    # create folder if not exist
    if not os.path.exists(reconstruction_path):
        print(f"Reconstruction path {reconstruction_path} does not exist, creating")
        os.makedirs(reconstruction_path, exist_ok=True)
    
    reconstruction = pycolmap.Reconstruction(reconstruction_path)
    print(f"Reconstruction summary:\n{reconstruction.summary()}")


    # Intrinsics
    cam_id, cam = next(iter(reconstruction.cameras.items()))
    fx, fy, cx, cy = cam.params
    width, height = cam.width, cam.height
    focal_px = float((fx + fy) / 2.0)
    camera_angle_x = float(2.0 * np.arctan(width / (2.0 * focal_px)))

    # Pick a registered image deterministically
    img_id = reconstruction.reg_image_ids()[0]      # list[int]
    img = reconstruction.image(img_id)              # pycolmap.Image

    # get points
    #points = np.array([pt.xyz for pt in reconstruction.points3D.values()], dtype=np.float32)  # (N, 3)
    # load ply points
    
    out_vggt = config.get("output_vggt", "../output/vggt/sparse")
    os.makedirs(out_vggt, exist_ok=True)

    points_path = os.path.abspath(os.path.join(out_vggt, "points.ply"))
    depth_path = os.path.abspath(os.path.join(out_vggt, "depth_map.png"))

    # Check if points.ply exists
    if not os.path.exists(points_path):
        raise FileNotFoundError(f"Points file not found at: {points_path}")

    # load ply points
    points = trimesh.load(points_path).vertices
    logging.info(f"Points (world coordinates) with shape and type: {points.shape}, {points.dtype}")

    # # World -> camera [R|t] from pycolmap (preferred API) : fix for 3.10 from 3.12
    cam_from_world = img.cam_from_world
    if callable(cam_from_world):  # >= 3.12
        T_cw = cam_from_world().matrix().astype(np.float32)
    else:  # <= 3.10
        T_cw = cam_from_world.matrix().astype(np.float32)

    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = T_cw[:, :3]
    extrinsic[:3, 3]  = T_cw[:, 3]

    # fix opencv/vggt and rotate to match blender coords
    R_fix = np.array([
    [1,  0,  0],
    [0,  0, -1],
    [0,  1,  0]
    ], dtype=np.float32)

    # extrinsic fix
    extrinsic[:3, :3] = R_fix @ extrinsic[:3, :3]
    extrinsic[:3, 3]  = R_fix @ extrinsic[:3, 3]

    # point fix
    R_p3d, T_p3d = B2P(extrinsic)
    # Transform points into the same space
    # (apply R_fix first, then B2P)
    points_fixed = (points @ R_fix.T)        # VGGT->Blender
    points_fixed = (points_fixed @ R_p3d.T)  # Blender->PyTorch3D
    points_fixed += T_p3d                   # Blender->PyTorch3D

    # flip points over x axis (back of cam to front)
    points_fixed[:, 1] *= -1
    # scale uniformly 
    points_fixed *= config.get("vggt_scene_scale", 5.0) # TODO: Check if camera needs scale as well

    # Assemble data
    camera_data = {
        "extrinsic": extrinsic,
        "focal": np.float32(focal_px),
        "image_size": np.array([width, height], dtype=np.int32),
        "camera_angle_x": np.float32(camera_angle_x),
    }

    path = os.path.abspath(config["camera"])
    # create dir from filename
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # print save to absolute path: 
    print(f"Saving camera data to: {path}")

    np.savez(path, **camera_data)
    print(f"Saved legacy camera file to: {path}")

    # save points to ply
    ply_path = os.path.abspath(config["vggt_cloud"])
    print(f"Saving points to: {ply_path}")
    # create dir
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)
    save_point_cloud(torch.tensor(points_fixed), ply_path, blender_readable=False)

    # Optional sanity‑check: load it back and print a few entries
    loaded = np.load(path)
    print("Loaded back – keys:", list(loaded.keys()))
    print("extrinsic[:3,:3] =\n", loaded["extrinsic"][:3, :3])
    print("focal (px) =", loaded["focal"])
    print("camera_angle_x (rad) =", loaded["camera_angle_x"])


#################################################################################

#################################################################################


# ✅
def prepare_images(images=None):
    if images is None:
        raise ValueError("No images provided for preparation.")

    logging.debug(f"Preparing images with shape: {images.shape}")

    if isinstance(images, list):
        images = torch.stack([torch.tensor(image) for image in images], dim=0)

    if len(images.shape) == 3:
        images = images.unsqueeze(0)  # Add batch dimension if missing

    if images.shape[1] == 4:  # If images are RGBA, convert to RGB
        images = images[:, :3, :, :]

    # if images are W,H switch to H,W (if H has < W)
    if images.shape[2] < images.shape[3]:
        images = images.permute(0, 1, 3, 2)

    logging.debug(f"Prepared images with shape: {images.shape}")

    return images

# ✅
def run_VGGT(model, images, dtype, resolution=518):
    # images: [B, 3, H, W]    
    images = prepare_images(images)
    
    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf


# ✅
def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction


# ✅
# ✅
def process_single_image_vggt(image_path, output_dir, model, device, dtype, config):
    """
    Process a single image with VGGT and save reconstruction to output_dir.
    
    Parameters
    ----------
    image_path: str
        Path to the input image
    output_dir: str
        Directory to save the reconstruction
    model: VGGT
        The VGGT model (already loaded and on device)
    device: torch.device
        Device to run on
    dtype: torch.dtype
        Data type for computation
    config: dict
        Configuration dictionary
        
    Returns
    -------
    bool
        True if successful
    str
        Path to the saved .ply file
    np.ndarray
        Extrinsic matrix (T_cw) from VGGT
    np.ndarray
        Intrinsic matrix from VGGT
    """
    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    images, original_coords = load_and_preprocess_images_square([image_path], img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    logging.info(f"Loaded images from {image_path}")

    # Run VGGT to estimate camera and depth
    # These are the raw camera parameters we need to return
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    if config.get("use_ba", False):
        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / vggt_fixed_resolution
        shared_camera = config.get("shared_camera", False)

        with torch.cuda.amp.autocast(dtype=dtype):
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                images,
                conf=depth_conf,
                points_3d=points_3d,
                masks=None,
                max_query_pts=config.get("max_query_pts", 4096),
                query_frame_num=config.get("query_frame_num", 8),
                keypoint_extractor="aliked+sp",
                fine_tracking=config.get("fine_tracking", True),
            )

            torch.cuda.empty_cache()

        intrinsic_ba = intrinsic.copy() # Keep original intrinsic for return
        intrinsic_ba[:, :2, :] *= scale
        track_mask = pred_vis_scores > config.get("vis_thresh", 0.2)

        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic_ba, # Use scaled intrinsic for BA
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=config.get("max_reproj_error", 8.0),
            shared_camera=shared_camera,
            camera_type=config.get("camera_type", "SIMPLE_PINHOLE"),
            points_rgb=points_rgb,
        )

        if reconstruction is None:
            raise ValueError("No reconstruction can be built with BA")

        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)

        reconstruction_resolution = img_load_resolution
    else:
        conf_thres_value = config.get("conf_thres_value", 5.0)
        max_points_for_colmap = config.get("max_points_for_colmap", 100000)
        shared_camera = False
        camera_type = "PINHOLE"

        image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d.shape

        points_rgb = F.interpolate(
            images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        conf_mask = depth_conf >= conf_thres_value
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        print("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )

        reconstruction_resolution = vggt_fixed_resolution

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        [image_path],
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving reconstruction to {os.path.abspath(output_dir)}")

    # Save depth map as image
    print("Depth map type and shape:", depth_map.dtype, depth_map.shape)
    depth_map = depth_map.squeeze()

    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    depth_path = os.path.join(output_dir, "depth_map.png")
    imageio.imwrite(depth_path, depth_uint8)
    print(f"Saved depth map to: {depth_path}")

    reconstruction.write(output_dir)

    # Save point cloud for fast visualization
    ply_filename = os.path.join(output_dir, "points.ply")
    trimesh.PointCloud(points_3d, colors=points_rgb).export(ply_filename)
    
    # ***MODIFIED RETURN***
    # Return the raw camera parameters *before* any BA or scaling
    return True, ply_filename, extrinsic, intrinsic


# ✅
def demo_fn(config):
    # Set seed for reproducibility
    seed = config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    print(f"Setting seed as: {seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Run VGGT for camera and depth estimation
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    print(f"Model loaded")

    # Get image paths and preprocess them
    image = config["image_url"]
    out_vggt = config.get("output_vggt", "../output/vggt/sparse")
    
    # Process the main image (CALL #1)
    # This now also returns the raw camera parameters
    print("=" * 80)
    print("Processing main image...")
    print("=" * 80)
    success, main_ply_path, main_ext, main_int = process_single_image_vggt(
        image, out_vggt, model, device, dtype, config
    )
    print(f"Main room point cloud saved to: {main_ply_path}")
    print(f"Captured main room camera parameters.")
    
    # Check if empty room image exists and process it
    empty_room_paths = [
        os.path.join("findings/banana/inpaint_nanoBanana", "empty_room.png"),
        os.path.join(config.get("output_inp_banana", ""), "empty_room.png")
    ]
    
    empty_room_image = None
    for path in empty_room_paths:
        if os.path.exists(path):
            empty_room_image = path
            break
    
    if empty_room_image:
        # --- NEW LOGIC: PROCESS EMPTY ROOM MANUALLY ---
        print("=" * 80)
        print(f"Found empty room image at: {empty_room_image}")
        print("Processing empty room image (depth map only)...")
        print("=" * 80)
        
        # 1. Define resolutions (must match process_single_image_vggt)
        vggt_fixed_resolution = 518
        img_load_resolution = 1024

        # 2. Load and preprocess the empty room image
        empty_images_tensor, _ = load_and_preprocess_images_square(
            [empty_room_image], img_load_resolution
        )
        empty_images_tensor = empty_images_tensor.to(device)

        # 3. Run VGGT (CALL #2), but we ONLY care about depth map and confidence
        # We DISCARD the extrinsic and intrinsic predicted for the empty room.
        print("Running VGGT for empty room depth map...")
        _empty_ext, _empty_int, empty_depth_map, empty_depth_conf = run_VGGT(
            model, empty_images_tensor, dtype, vggt_fixed_resolution
        )
        
        # 4. UNPROJECT empty depth map using MAIN room's camera parameters
        print("Unprojecting empty depth map using MAIN room's camera...")
        empty_points_3d = unproject_depth_map_to_point_map(
            empty_depth_map, main_ext, main_int # <-- Using main cam params
        )

        # 5. Get colors for the empty room point cloud
        # (This logic is from the 'else' block of process_single_image_vggt)
        empty_points_rgb = F.interpolate(
            empty_images_tensor, size=(vggt_fixed_resolution, vggt_fixed_resolution), 
            mode="bilinear", align_corners=False
        )
        empty_points_rgb = (empty_points_rgb.cpu().numpy() * 255).astype(np.uint8)
        empty_points_rgb = empty_points_rgb.transpose(0, 2, 3, 1) # (B, H, W, 3)

        # 6. Filter points by confidence
        # (This logic is from the 'else' block of process_single_image_vggt)
        conf_thres_value = config.get("conf_thres_value", 5.0)
        max_points_for_colmap = config.get("max_points_for_colmap", 100000)
        
        conf_mask = empty_depth_conf >= conf_thres_value
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)
        
        # empty_points_3d is (1, H, W, 3), conf_mask is (1, H, W)
        empty_points = empty_points_3d[conf_mask]
        empty_colors = empty_points_rgb[conf_mask]
        
        print(f"Unprojected empty room to {empty_points.shape[0]} points.")

        # 7. Save this (unaligned) point cloud
        empty_room_ply_final = os.path.join(out_vggt, "points_emptyRoom.ply")
        trimesh.PointCloud(empty_points, colors=empty_colors).export(empty_room_ply_final)
        print(f"Empty room point cloud (unprojected w/ main cam) saved to: {empty_room_ply_final}")

        # 8. Load the main room's point cloud (from the file) for alignment
        print("Loading main room's final point cloud for alignment...")
        main_pc = trimesh.load(main_ply_path)
        main_points = main_pc.vertices
        
        # --- END NEW LOGIC ---
        
        # 9. Align empty room point cloud to main room point cloud
        # This should now work, as they are in the same coordinate system
        print("=" * 80)
        print("Aligning empty room to main room (scale + translation only)...")
        print("=" * 80)
        
        # We already have main_points and empty_points in memory
        
        # Debug: Print original statistics
        print(f"Before alignment:")
        print(f"  Main room - center: {np.mean(main_points, axis=0)}, extents: {np.max(main_points, axis=0) - np.min(main_points, axis=0)}")
        print(f"  Empty room - center: {np.mean(empty_points, axis=0)}, extents: {np.max(empty_points, axis=0) - np.min(empty_points, axis=0)}")
        print(f"  Main room bounds: min={np.min(main_points, axis=0)}, max={np.max(main_points, axis=0)}")
        print(f"  Empty room bounds: min={np.min(empty_points, axis=0)}, max={np.max(empty_points, axis=0)}")
        
        # Align empty room to main room using OBB matching
        aligned_empty_points, scale, R, t = align_pointclouds_obb(empty_points, main_points)
        
        print(f"\nAlignment applied:")
        print(f"  Translation: {t}")
        
        # Debug: Print aligned statistics
        print(f"\nAfter alignment:")
        print(f"  Aligned empty room - center: {np.mean(aligned_empty_points, axis=0)}, extents: {np.max(aligned_empty_points, axis=0) - np.min(aligned_empty_points, axis=0)}")
        print(f"  Aligned empty room bounds: min={np.min(aligned_empty_points, axis=0)}, max={np.max(aligned_empty_points, axis=0)}")
        
        # Check if alignment actually changed anything
        diff = np.abs(aligned_empty_points - empty_points).max()
        print(f"  Max point difference: {diff:.6f}")
        
        # Save aligned empty room point cloud
        aligned_empty_pc = trimesh.PointCloud(aligned_empty_points, colors=empty_colors)
        aligned_empty_ply_path = os.path.join(out_vggt, "points_emptyRoom_aligned.ply")
        aligned_empty_pc.export(aligned_empty_ply_path)
        print(f"\nAligned empty room point cloud saved to: {aligned_empty_ply_path}")
        print(f"Original unaligned empty room kept at: {empty_room_ply_final}")
    else:
        print("No empty room image found, skipping empty room processing")

    return True


# ✅
if __name__ == "__main__":
    # Load configuration
    parser = argparse.ArgumentParser(description="Run segmentation script with config file.")
    parser.add_argument("--config", default="../src/config.yaml", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)

    debug_level = config.get("logging", "DEBUG")
    logging.basicConfig(level=debug_level)

    with torch.no_grad():
        demo_fn(config)
        export_vggt_data(config)