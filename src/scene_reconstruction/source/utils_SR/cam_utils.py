
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

from pytorch3d.renderer import PerspectiveCameras

def calibrate_cameras(config: dict, width: int, height: int, device: torch.device, **kwargs):
    # 1. Load Data
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

    # 2. Extrinsics
    ext = torch.from_numpy(data["extrinsic"]).float()
    R, T = B2P(ext.numpy())
    R = torch.from_numpy(R).float().to(device).unsqueeze(0)
    T = torch.from_numpy(T).float().to(device).unsqueeze(0)

    # 3. Setup Screen-Space Parameters (The "Image Factor" approach)
    # -------------------------------------------------------------------------
    orig_w, orig_h = data["image_size"]
    focal_px_orig = float(data["focal"])
    
    # Scale focal length to match the current render resolution.
    # We assume Vertical FOV is consistent (standard in Blender/Unity).
    # If your setup locks Horizontal FOV, change this to: width / orig_w
    scale = height / orig_h 
    focal_px_new = focal_px_orig * scale

    # Define Focal Length in Pixels (Square pixels imply fx == fy)
    # We use Float tensors to match the pointcloud precision
    focal_length_screen = torch.tensor([[focal_px_new, focal_px_new]], device=device).float()
    
    # Define Principal Point in Pixels (Center of the image)
    # This is much safer than assuming (0,0) in NDC
    pp_x = width / 2.0
    pp_y = height / 2.0
    principal_point_screen = torch.tensor([[pp_x, pp_y]], device=device).float()
    
    # -------------------------------------------------------------------------
    
    near_clip = config.get("camera_znear", 0.1)
    far_clip = config.get("camera_zfar", 100.0)

    # 4. Create Camera with in_ndc=False
    cameras = PerspectiveCameras(
        device=device,
        focal_length=focal_length_screen,
        principal_point=principal_point_screen,
        R=R,
        T=T,
        in_ndc=False,  # <--- This tells PyTorch3D "My parameters are in Pixels"
        image_size=((height, width),), # (H, W) is required for the internal conversion
    )

    return cameras, R, T, focal_px_new

# def calibrate_cameras(config: dict, width: int, height: int, device: torch.device, **kwargs):
#     # Load camera data from .npz

#     if config.get("use_3d_front", False):
#         logging.info("Using 3D-FRONT camera parameters.")
#         data = np.load(
#             extract_camera_from_json(
#                 image_path=config["input_image"], output_folder=config["tmp_dir"]
#             )
#         )
#     else:
#         data = np.load(config["camera"])

#     if config.get("Use_VGGT", False):
#         logging.debug("Using VGGT camera parameters.")
#         data = np.load(config["camera"])

#     # Extract extrinsic matrix (4x4) from Blender
#     ext = torch.from_numpy(data["extrinsic"]).float()  # Shape: (4,4)

#     R, T = B2P(ext.numpy())  # Convert to PyTorch3D format

#     R = torch.from_numpy(R).float().to(device).unsqueeze(0)  # Shape: (1, 3, 3)
#     T = torch.from_numpy(T).float().to(device).unsqueeze(0)  # Shape: (1, 3)

#     focal_px = float(data["focal"])
#     logging.debug(f"Camera focal length from JSON: {focal_px:.2f} px")


#     orig_w, orig_h = data["image_size"]        # check dataset ordering; many datasets give (width, height)
#     focal_px = float(data["focal"])            # original focal in pixels
#     render_w = width
#     render_h = height

#     logging.debug(f"Original image size from JSON: {orig_w} x {orig_h}")
#     logging.debug(f"Render image size: {render_w} x {render_h}")
#     logging.debug(f"Camera focal length from JSON: {focal_px:.2f} px")

#     focal_px_scaled = focal_px * (render_h / orig_h)

#     # compute vertical FoV (PyTorch3D expects vertical fov for FoV cameras)
#     fov = 2.0 * np.arctan(render_h / (2.0 * focal_px_scaled))  # radians TODO : check h or w

#     # kwargs nearclip and farclip
#     near_clip = config.get("camera_znear", 0.1)
#     far_clip = config.get("camera_zfar", 100.0)
#     # Create PyTorch3D camera
#     cameras = FoVPerspectiveCameras(
#         device=device,
#         fov=float(fov),  # fov=fov,
#         degrees=False,
#         #aspect_ratio=render_w / render_h,
#         R=R,  # Add batch dimension
#         T=T,  # Add batch dimension
#         znear=near_clip,
#         zfar=far_clip,
#     )

#     return cameras, R, T, focal_px