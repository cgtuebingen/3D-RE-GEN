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

def align_pointclouds_pca(source_points: np.ndarray, target_points: np.ndarray):
    """
    PCA-based rigid alignment of source to target.
    Returns aligned source points, rotation R (3x3), translation t (3,).
    """
    if source_points.ndim != 2 or target_points.ndim != 2 or source_points.shape[1] != 3 or target_points.shape[1] != 3:
        raise ValueError("Points must be (N,3) and (M,3)")

    # Centers
    src_c = np.mean(source_points, axis=0)
    tgt_c = np.mean(target_points, axis=0)

    src_centered = source_points - src_c
    tgt_centered = target_points - tgt_c

    # Principal axes via SVD of centered data
    _, _, Vt_src = np.linalg.svd(src_centered, full_matrices=False)
    _, _, Vt_tgt = np.linalg.svd(tgt_centered, full_matrices=False)
    axes_src = Vt_src  # rows: principal axes
    axes_tgt = Vt_tgt

    # Rotation from source axes to target axes
    R = axes_tgt.T @ axes_src
    # Ensure right-handed rotation (det=+1)
    if np.linalg.det(R) < 0:
        axes_tgt[-1, :] *= -1
        R = axes_tgt.T @ axes_src

    # Apply transform
    aligned_centered = src_centered @ R.T
    aligned = aligned_centered + tgt_c
    t = tgt_c - (src_c @ R.T)
    return aligned, R, t




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


    # Helper to extract camera intrinsics info for a given pycolmap image
    def _intrinsics_for_image(pyimg: pycolmap.Image):
        pycam = reconstruction.cameras[pyimg.camera_id]
        fx, fy, cx, cy = pycam.params
        width, height = pycam.width, pycam.height
        focal_px = float((fx + fy) / 2.0)
        camera_angle_x = float(2.0 * np.arctan(width / (2.0 * focal_px)))
        return focal_px, width, height, camera_angle_x

    # Pick the camera corresponding to the first input image (frame 0)
    target_image_name = None
    img_list_path = os.path.join(reconstruction_path, "image_list.txt")
    if os.path.exists(img_list_path):
        try:
            with open(img_list_path, "r") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            if lines:
                target_image_name = lines[0]
        except Exception as e:
            print(f"Warning: could not read image_list.txt: {e}")

    # Find image by exact name or basename match; fallback to first registered
    img = None
    if target_image_name is not None:
        target_base = os.path.basename(target_image_name)
        for image_id, pyimg in reconstruction.images.items():
            if pyimg.name == target_image_name or os.path.basename(pyimg.name) == target_base:
                img = pyimg
                break

    if img is None:
        # Fallback deterministic selection
        img_id = reconstruction.reg_image_ids()[0]
        img = reconstruction.image(img_id)

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

    # World -> camera [R|t] for main image
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

    # Assemble data for main image
    focal_px, width, height, camera_angle_x = _intrinsics_for_image(img)
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

    # If a second image (empty room) exists, also export its camera as camera_emptyRoom.npz
    empty_img = None
    if os.path.exists(img_list_path):
        try:
            with open(img_list_path, "r") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            if len(lines) >= 2:
                empty_name = lines[1]
                empty_base = os.path.basename(empty_name)
                for _, pyimg in reconstruction.images.items():
                    if pyimg.name == empty_name or os.path.basename(pyimg.name) == empty_base:
                        empty_img = pyimg
                        break
        except Exception as e:
            print(f"Warning: could not read image_list.txt for empty camera: {e}")

    if empty_img is not None:
        cam_from_world2 = empty_img.cam_from_world
        if callable(cam_from_world2):
            T_cw2 = cam_from_world2().matrix().astype(np.float32)
        else:
            T_cw2 = cam_from_world2.matrix().astype(np.float32)

        extrinsic2 = np.eye(4, dtype=np.float32)
        extrinsic2[:3, :3] = T_cw2[:, :3]
        extrinsic2[:3, 3]  = T_cw2[:, 3]

        # Apply the same coordinate fix as for main camera
        extrinsic2[:3, :3] = R_fix @ extrinsic2[:3, :3]
        extrinsic2[:3, 3]  = R_fix @ extrinsic2[:3, 3]

        focal2, w2, h2, camang2 = _intrinsics_for_image(empty_img)
        camera_empty = {
            "extrinsic": extrinsic2,
            "focal": np.float32(focal2),
            "image_size": np.array([w2, h2], dtype=np.int32),
            "camera_angle_x": np.float32(camang2),
        }

        empty_path = os.path.join(os.path.dirname(path), "camera_emptyRoom.npz")
        np.savez(empty_path, **camera_empty)
        print(f"Saved empty-room camera file to: {empty_path}")

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
    # if image is single make a list
    if not isinstance(image_path, list):
        image_path = [image_path]

    images, original_coords = load_and_preprocess_images_square(image_path, img_load_resolution)
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

        # Keep copies for per-frame saving before flattening for COLMAP
        points_3d_full = points_3d
        points_rgb_full = points_rgb
        conf_mask_full = conf_mask

        # Flatten for COLMAP reconstruction input
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
        image_path,
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

    # depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    # depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    # depth_path = os.path.join(output_dir, "depth_map.png")
    # imageio.imwrite(depth_path, depth_uint8)
    # print(f"Saved depth map to: {depth_path}")

    reconstruction.write(output_dir)

    # Persist the ordered image list so downstream can pick the correct camera (frame 0)
    try:
        img_list_path = os.path.join(output_dir, "image_list.txt")
        with open(img_list_path, "w") as f:
            for p in image_path:
                f.write(str(p) + "\n")
    except Exception as e:
        print(f"Warning: could not save image_list.txt: {e}")

    # Save point clouds
    # 1) Merged (backward-compatible)
    ply_filename = os.path.join(output_dir, "points_merged.ply")
    trimesh.PointCloud(points_3d, colors=points_rgb).export(ply_filename)
    
    # 2) Per-frame clouds if multiple frames (e.g., main + background)
    try:
        num_frames = points_3d_full.shape[0]
        if num_frames >= 1:
            # frame 0: main
            mask0 = conf_mask_full[0]
            pts0 = points_3d_full[0][mask0]
            rgb0 = points_rgb_full[0][mask0]
            main_ply = os.path.join(output_dir, "points.ply")
            trimesh.PointCloud(pts0, colors=rgb0).export(main_ply)
            # Prefer returning main_ply path if created
            ply_filename = main_ply
            # frame 1: background (if present)
            if num_frames >= 2:
                mask1 = conf_mask_full[1]
                pts1 = points_3d_full[1][mask1]
                rgb1 = points_rgb_full[1][mask1]
                # Save original background as _pre
                bg_ply_pre = os.path.join(output_dir, "points_emptyRoom_pre.ply")
                trimesh.PointCloud(pts1, colors=rgb1).export(bg_ply_pre)

                # Easy world-space fit: per-axis scale + translate (no rotation)
                # Match extents of background (source) to main (target)
                bg_ply_fit = os.path.join(output_dir, "points_emptyRoom.ply")
                if pts1.size == 0 or pts0.size == 0:
                    # Fallback: just save raw if either set is empty
                    trimesh.PointCloud(pts1, colors=rgb1).export(bg_ply_fit)
                else:
                    # Scale-only: match bbox sizes per-axis; no translation
                    # Compute extents directly (translation-invariant)
                    src_min, src_max = np.min(pts1, axis=0), np.max(pts1, axis=0)
                    tgt_min, tgt_max = np.min(pts0, axis=0), np.max(pts0, axis=0)
                    src_ext = src_max - src_min
                    tgt_ext = tgt_max - tgt_min
                    # Per-axis scale with safe divide
                    scale = np.divide(tgt_ext, src_ext, out=np.ones_like(tgt_ext), where=src_ext > 1e-6)
                    # Apply scale about source centroid to avoid shifting center
                    src_c = np.mean(pts1, axis=0)
                    pts1_fit = (pts1 - src_c) * scale + src_c
                    trimesh.PointCloud(pts1_fit, colors=rgb1).export(bg_ply_fit)
    except Exception as e:
        print(f"Per-frame point cloud export skipped due to error: {e}")
    
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

    # Get image paths and (optionally) include background as second frame
    image_main = config["image_url"]
    image_bg = "../output/findings/banana/inpaint_nanoBanana/empty_room.png"
    if os.path.exists(image_bg):
        print(f"Found background image at: {image_bg}")

        # resize to same size as main image
        from PIL import Image
        img_main_pil = Image.open(image_main)
        img_bg_pil = Image.open(image_bg)
        img_bg_pil = img_bg_pil.resize(img_main_pil.size, Image.LANCZOS)
        image_bg_resized = os.path.join(os.path.dirname(image_bg), "empty_room.png")
        img_bg_pil.save(image_bg_resized)
        image_bg = image_bg_resized
        print(f"Resized background image saved to: {image_bg}")

        images = [image_main, image_bg]
    else:
        images = [image_main]

    out_vggt = config.get("output_vggt", "../output/vggt/sparse")

    # Single run; process and save into sparse as before (per-frame PLYs handled inside)
    print("=" * 80)
    print("Processing inputs with VGGT...")
    print("=" * 80)
    success, main_ply_path, main_ext, main_int = process_single_image_vggt(
        images, out_vggt, model, device, dtype, config
    )
    print(f"Point clouds saved under: {out_vggt}")
    print(f"Main room point cloud saved to: {main_ply_path}")
    print(f"Captured main room camera parameters.")
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