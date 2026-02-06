
from pytorch3d.renderer import (
    FoVPerspectiveCameras
)

from scene_reconstruction.source.utils_SR.diff_utils import (

    extract_camera_from_json,

)

import sys
import torch
import torch
import numpy as np


sys.path.insert(0, "../")
from utils.global_utils import (

    B2P,

)
import logging

def calibrate_cameras(config: dict, width: int, height: int, device: torch.device, **kwargs):
    # Load camera data from .npz

    if config.get("use_3d_front", False):
        logging.info("Using 3D-FRONT camera parameters.")
        data = np.load(
            extract_camera_from_json(
                image_path=config["input_image"], output_folder=config["tmp_dir"]
            )
        )
    else:
        data = np.load(config["camera"])

    if config.get("Use_VGGT", False):
        logging.debug("Using VGGT camera parameters.")
        data = np.load(config["camera"])

    # Extract extrinsic matrix (4x4) from Blender
    ext = torch.from_numpy(data["extrinsic"]).float()  # Shape: (4,4)

    R, T = B2P(ext.numpy())  # Convert to PyTorch3D format

    R = torch.from_numpy(R).float().to(device).unsqueeze(0)  # Shape: (1, 3, 3)
    T = torch.from_numpy(T).float().to(device).unsqueeze(0)  # Shape: (1, 3)

    focal_px = float(data["focal"])
    logging.debug(f"Camera focal length from JSON: {focal_px:.2f} px")


    orig_w, orig_h = data["image_size"]        # check dataset ordering; many datasets give (width, height)
    focal_px = float(data["focal"])            # original focal in pixels
    render_w = width
    render_h = height

    logging.debug(f"Original image size from JSON: {orig_w} x {orig_h}")
    logging.debug(f"Render image size: {render_w} x {render_h}")
    logging.debug(f"Camera focal length from JSON: {focal_px:.2f} px")

    focal_px_scaled = focal_px * (render_h / orig_h)

    # compute vertical FoV (PyTorch3D expects vertical fov for FoV cameras)
    fov = 2.0 * np.arctan(render_h / (2.0 * focal_px_scaled))  # radians TODO : check h or w

    # kwargs nearclip and farclip
    near_clip = config.get("camera_znear", 0.1)
    far_clip = config.get("camera_zfar", 100.0)
    # Create PyTorch3D camera
    cameras = FoVPerspectiveCameras(
        device=device,
        fov=float(fov),  # fov=fov,
        degrees=False,
        #aspect_ratio=render_w / render_h,
        R=R,  # Add batch dimension
        T=T,  # Add batch dimension
        znear=near_clip,
        zfar=far_clip,
    )

    return cameras, R, T, focal_px