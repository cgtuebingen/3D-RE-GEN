import random
import sys
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F
import yaml

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
# add debugger
import logging

#### Added ##################################################################

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
    points_path = os.path.join(out_vggt, "points.ply")
    depth_path = os.path.join(out_vggt, "depth_map.png")

    # get images 

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
    image =  config["image_url"] #os.path.join(config["scene_dir"], "images")

    vggt_fixed_resolution = 518
    img_load_resolution = config.get("image_size_DR", 1024)

    images, original_coords = load_and_preprocess_images_square([image], img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    logging.info(f"Loaded images from {image}")

    # Run VGGT to estimate camera and depth
    # Run with 518x518 images
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    if config.get("use_ba", False):
        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / vggt_fixed_resolution
        shared_camera = config.get("shared_camera", False)

        with torch.cuda.amp.autocast(dtype=dtype):
            # Predicting Tracks
            # Using VGGSfM tracker instead of VGGT tracker for efficiency
            # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
            # Will be fixed in VGGT v2

            # You can also change the pred_tracks to tracks from any other methods
            # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
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

        # rescale the intrinsic matrix from 518 to 1024
        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > config.get("vis_thresh", 0.2)

        # TODO: radial distortion, iterative BA, masks
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
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

        # Bundle Adjustment
        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)

        reconstruction_resolution = img_load_resolution
    else:
        conf_thres_value = config.get("conf_thres_value", 5.0)
        max_points_for_colmap = config.get("max_points_for_colmap", 100000)  # randomly sample 3D points
        shared_camera = False  # in the feedforward manner, we do not support shared camera
        camera_type = "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera

        image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d.shape

        points_rgb = F.interpolate(
            images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        # (S, H, W, 3), with x, y coordinates and frame indices
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        conf_mask = depth_conf >= conf_thres_value
        # at most writing 100000 3d points to colmap reconstruction object
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
        [image],
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    out_vggt = config.get("output_vggt", "../output/vggt/sparse")
    logging.info(f"Saving reconstruction to {os.path.abspath(out_vggt)}")

    # save out depth map
    import imageio
    
    # Save depth map as image
    print("Depth map type and shape:", depth_map.dtype, depth_map.shape)
    # from [1,x,x,1] to [x,x]
    depth_map = depth_map.squeeze()

    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    depth_path = os.path.join(out_vggt, "depth_map.png")
    imageio.imwrite(depth_path, depth_uint8)
    print(f"Saved depth map to: {depth_path}")


    os.makedirs(out_vggt, exist_ok=True)
    reconstruction.write(out_vggt)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(out_vggt, "points.ply"))

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