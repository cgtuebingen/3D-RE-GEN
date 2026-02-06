
from utils.global_utils import (
    load_config,
    clear_output_directory,
    B2P,
    save_point_cloud
)
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../src/scene_reconstruction/source")))
from utils_SR.cam_utils import (
    calibrate_cameras
)
from utils_SR.pc_utils import (
    get_model_vggt_cloud,
    filter_points_by_quantile,
    filter_dbscan
)

import argparse
import torch
#import open3d as o3d

from trimesh import load

import imageio.v2 as imageio
import numpy as np
import os.path
import logging

from cv2 import resize
from tqdm import tqdm

from open3d import (
    geometry,
    utility
)
print("Done loading imports.")

def pointcloud_prep(target_pointcloud, device)->torch.Tensor:
    if isinstance(target_pointcloud, torch.Tensor):
        tp = target_pointcloud
        if tp.dim() == 3 and tp.shape[0] == 1:
            tp = tp.squeeze(0)
    else:
        tp = torch.as_tensor(target_pointcloud, dtype=torch.float32, device=device)

    tp = tp.to(device=device, dtype=torch.float32)
    if tp.dim() != 2 or tp.shape[1] != 3:
        try:
            tp = tp.reshape(-1, 3)
        except Exception:
            raise RuntimeError(
                f"target_pointcloud has invalid shape: {tuple(tp.shape)}"
            )
    return tp


def main(config: dict, model_name, mask_folder="../output/masks", point_cloud_folder="../output/pointclouds/"):
    # config parameters
    device = config["device"]
    torch.manual_seed(config.get("seed", 12345))
    image_size = config.get("image_size_DR", 512)


    ############### Mask Generation ###############

    # Load image and save out masks
    image_path = config.get("image_url", None)
    if image_path is None:
        raise ValueError("Image path not provided in config.")

    # "../../2D_to_3D_augmentor/intertior_test.jpg")
    image = imageio.imread(image_path)
    # image = image.convert("RGB")
    image_size_orig = image.shape[:2]

       # TODO load full size image path
    model_image_path = config.get("full_size", None)

    if model_image_path is None:
        raise ValueError("Model image path not provided in config.")

    # add model name to path
    model_image_path = os.path.join(model_image_path, model_name + ".png")

    # "../../2D_to_3D_augmentor/findings/fullSize/furniture__(810, 861).png"

    image_ref = imageio.imread(model_image_path)  # load non cropped image of object
    # save size dimesions of the image
    orig_height, orig_width = image_size_orig
    logging.debug("Original image size: %d x %d", orig_width, orig_height)

    scale_factor = image_size / orig_width  # scale factor for y
    target_height = int(orig_height * scale_factor)

    # Resize the reference image to the desired size, relative
    image_ref = resize(image_ref, (image_size, target_height))
    logging.debug("Reference image resized to: %s", image_ref.shape)

    # Normalize image_ref to [0, 1] if needed and create mask
    if image_ref.max().item() > 1:
        image_ref_norm = image_ref.astype(np.float32) / 255.0
    else:
        image_ref_norm = image_ref.copy()

    # Compute the mask: pixels that are not pure white (mean across RGB channels less than 0.99)
    mask = (np.mean(image_ref_norm[..., :3], axis=-1) < 0.99).astype(np.float32)




    #### END Mask Generation ####

    import cv2

    # shrink_pixels: number of pixels to shrink the mask by (hyperparameter)
    shrink_pixels = config.get("mask_shrink_pixels", 3)
    if shrink_pixels > 0:
        kernel = np.ones((shrink_pixels, shrink_pixels), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=config.get("mask_shrink_iterations", 2))
    
    # Save mask - convert to uint8 for PNG format
    mask_path = os.path.join(mask_folder, f"{model_name}.png")
    mask_uint8 = (mask * 255).astype(np.uint8)
    imageio.imwrite(mask_path, mask_uint8)

    ############### Point Cloud Extraction ##################################################################
    # Calibrate cameras
    cameras, R, T, focal_px = calibrate_cameras(
       config=config, device=device, width=image_size, height=target_height
    )

    print("Camera size:", image_size, target_height)

    # # Create point cloud
    # pc_trimesh = load(config["vggt_cloud"])
    # raw_pc = torch.from_numpy(pc_trimesh.vertices).float().to(device)

    # R_np, t_np = B2P(np.eye(4))
    # R_torch = torch.from_numpy(R_np).float().to(device)
    # t_torch = torch.from_numpy(t_np).float().to(device)
    # # Apply to the raw point cloud
    # raw_pc = raw_pc @ R_torch.T + t_torch

    # raw_pc[:, 1] *= -1

    # logging.debug(f"Raw VGGT cloud loaded: {raw_pc.shape[0]} points")

    # # Ensure proper shape (N, 3)
    # if raw_pc.dim() == 3 and raw_pc.shape[0] == 1:
    #     raw_pc = raw_pc.squeeze(0)
    # raw_pc = raw_pc.reshape(-1, 3)

    

    # Extract cropped point cloud from VGGT
    target_pointcloud = get_model_vggt_cloud(
        mask=mask,
        vggt_cloud_path=config["vggt_cloud"],
        cameras=cameras,
        device=device
    )

    logging.debug("Imported vggt cloud")

    # filter with quantiles
    if config.get("filter_vggt_quantile", False):
        target_pointcloud = filter_points_by_quantile(target_pointcloud, q=config.get("quantile_value", 0.05))
    # filter DBSCAN
    if config.get("filter_vggt_dbscan", True):
        target_pointcloud = filter_dbscan(target_pointcloud, eps=config.get("dbscan_eps", 0.035), min_samples=config.get("dbscan_min_points", 10))

    logging.debug("Filtered vggt cloud")

    # TODO : ADD open3d normal estimation for mesh to point loss in MODEL! 
    # tensor to numpy
    target_pointcloud = target_pointcloud.cpu().numpy()

    target_pointcloud = np.ascontiguousarray(target_pointcloud, dtype=np.float64)

    logging.debug("Shape before normals:", target_pointcloud.shape)
    # check for nan
    if np.isnan(target_pointcloud).any():
        logging.warning(f"  Skipping pointcloud with NaN values")
        return False

    pcd = geometry.PointCloud()
    pcd.points = utility.Vector3dVector(target_pointcloud)


    if len(pcd.points) == 0:
        logging.warning(f"  Skipping empty pointcloud")
        return False
    
    logging.debug(f"  Points: {len(pcd.points)}")
    
    pcd.normals = utility.Vector3dVector(np.zeros(
        (1, 3)))  # invalidate existing normals

    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)

    logging.debug("Estimated normals: ")

    # pcd back to numpy and tensor with normals
    target_pointcloud = np.asarray(pcd.points)
    target_normals = np.asarray(pcd.normals)
    target_pointcloud = torch.from_numpy(target_pointcloud).float().to(device)
    target_normals = torch.from_numpy(target_normals).float().to(device)

    logging.debug("Shape with normals:", target_pointcloud.shape)


    target_pointcloud = pointcloud_prep(target_pointcloud, device)
    target_normals = pointcloud_prep(target_normals, device)

    # save pointcloud
    point_out_path = os.path.join(point_cloud_folder, f"{model_name}.ply")
    save_point_cloud(target_pointcloud, point_out_path, blender_readable=False)

    # save normals
    normal_out_path = os.path.join(point_cloud_folder, "normals", f"{model_name}_normals.ply")
    os.makedirs(os.path.dirname(normal_out_path), exist_ok=True)
    save_point_cloud(target_normals, normal_out_path, blender_readable=False)

    ###### END Point Cloud Extraction ######




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Differential Rendering Runner")

    parser.add_argument(
        "--config",
        type=str,
        default="../src/config.yaml",
        help="Path to config file in YAML format",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    debug_level = config.get("logging", "DEBUG")
    logging.basicConfig(level=debug_level)

    # get "cropped" folder path from config
    cropped_folder = config.get("full_size", None)
    if cropped_folder is None:
        raise ValueError("Cropped folder path not found in config file.")

    mask_folder = config.get("mask_folder", "../output/masks")
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)
    clear_output_directory(mask_folder)

    point_cloud_folder = config.get("output_ply", "../output/pointclouds/")
    if not os.path.exists(point_cloud_folder):
        os.makedirs(point_cloud_folder)
    clear_output_directory(point_cloud_folder)

    # --- Prepare list of tasks ---
    # use tqdm for progress bar
    for image_name_ext in tqdm(os.listdir(cropped_folder)):

        image_name = os.path.splitext(image_name_ext)[0]
        main(config, image_name, mask_folder=mask_folder, point_cloud_folder=point_cloud_folder)