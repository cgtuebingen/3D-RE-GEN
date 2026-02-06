
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    look_at_rotation,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    BlendParams,
    SoftSilhouetteShader,
    HardPhongShader,
    PointLights,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor,
    SoftPhongShader,
    PointsRasterizationSettings,
)

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
from pytorch3d.io import IO
from pytorch3d.io import load_obj
from scipy import io
from scene_reconstruction.source.utils_SR.diff_utils import (
    normalized_to_camera_space,
    dice_loss,
    extract_camera_from_json,
    camera_to_world_space,
    #regularize_depth_map,
    clean_mesh,
    sample_mesh_points,
    depth_from_image,
    visualize_plane_and_axes,
    save_glb_mesh,
    visualize_pointclouds
)

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
    B2P,
    P2B,
)


import logging
import warnings


from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import euler_angles_to_matrix, Transform3d
import torch.nn.functional as F


def initialize_renderer(
    cameras: FoVPerspectiveCameras,
    sigma: float,
    gamma: float,
    image_size: int,
    target_height: int,
    device: torch.device,
):
    # Throw error if cameras is None
    if cameras is None:
        raise ValueError("Cameras not provided")

    # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of
    # edges. Refer to blending.py for more details.
    blend_params = BlendParams(sigma=sigma, gamma=gamma)

    # check for aspect ratio
    #print()

    raster_settings = RasterizationSettings(
        image_size=(target_height, image_size),
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
        # bin_size=0,
        faces_per_pixel=20,
        max_faces_per_bin=100000,  # increased to mitigate overflow
    )

    # Create a silhouette mesh renderer by composing a rasterizer and a shader.
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )

    lights = PointLights(
        device=device, location=((2.0, 2.0, -2.0),)
    )  # TODO: implement ENV light isntead of pointlights

    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights),
    )

    return silhouette_renderer, phong_renderer, raster_settings


def make_pointcloud_renderer(cameras, image_size, target_height, device):
    """
    Returns a renderer that draws a point cloud as small discs.
    """

    raster_settings = PointsRasterizationSettings(
        image_size=(target_height, image_size), radius=0.003, points_per_pixel=10, max_points_per_bin=100000
    )

    rasterizer = PointsRasterizer(
        cameras=cameras,
        raster_settings=raster_settings,
    )

    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(),  # just blend the RGBA values from the shader
    )
    return renderer