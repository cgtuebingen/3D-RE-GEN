import torch
import numpy as np
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

import os
import json


from typing import Tuple

from pytorch3d.transforms import euler_angles_to_matrix, Transform3d
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import (
    MeshRenderer,
    FoVPerspectiveCameras,
    TexturesVertex,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    look_at_view_transform,

)

from transformers import pipeline
from PIL import Image

import cv2
import trimesh

def lerp(a, b, bias):
    return a * (1 - bias) + b * bias

def load_ply_points(ply_path: str) -> np.ndarray:
    """Load (N,3) points from a PLY file using trimesh."""
    mesh = trimesh.load(ply_path, process=False)
    return np.asarray(mesh.vertices, dtype=np.float32)

def save_glb_mesh(glb_dir: str, model_name: str, glb_object: Meshes):
    """
    Save the mesh in GLB format to the specified directory with the model name.
    Args:
        glb_dir (str): Directory where the GLB file will be saved.
        model_name (str): Name of the model to be used in the GLB file name.
        glb_object (Meshes): The mesh object to be saved.
    """

    io = IO()
    io.register_meshes_format(MeshGlbFormat())

    try:
        glb_path = os.path.join(glb_dir, f"{model_name}.glb")

        io.save_mesh(data=glb_object.detach(), path=glb_path, include_textures=True)

        logging.info(
            f"######### Mesh saved successfully to {glb_path} #################"
        )
    except Exception as e:
        logging.error(f"Error saving mesh: {e}")


def depth_from_image(image_path) -> np.ndarray:
    """    
    Uses the Depth Anything model to estimate depth from an image.
    """
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf"
    )
    image = Image.open(image_path)
    image = image.convert("RGB")

    # inference
    depth_scene = pipe(image)["depth"]

    # Convert the depth image to a numpy array
    depth_scene = np.array(depth_scene)
    # invert depth values
    depth_scene = np.max(depth_scene) - depth_scene

    return depth_scene



def visualize_pointclouds(
    target_pts: torch.Tensor,  # (N, 3) tensor - NOT Pointclouds objects
    mesh: Meshes,
    num_samples: int,
    pc_renderer=None,
    config: dict = None,
    device="cpu",
    camera_distance=2.0,
    image_size_x=None,
    image_size_y=None,
):
    """
    Visualize two raw point tensors by wrapping them in Pointclouds ONLY for rendering.

    Args:
        target_pts: (N, 3) tensor of points (world coordinates)
        mesh_pts: (M, 3) tensor of points (world coordinates)
    """
    # Ensure both are 2D tensors (N, 3)
    target_pts = target_pts.squeeze() if target_pts.dim() > 2 else target_pts

    mesh_pts = sample_mesh_points(mesh=mesh, num_samples=num_samples)

    mesh_pts = mesh_pts.squeeze() if mesh_pts.dim() > 2 else mesh_pts

    # Create colored features (still tensors)
    red_feat = torch.ones_like(target_pts) * torch.tensor(
        [1.0, 0.0, 0.0], device=device
    )
    green_feat = torch.ones_like(mesh_pts) * torch.tensor(
        [0.0, 1.0, 0.0], device=device
    )

    # Combine into single point set for rendering
    combined_pts = torch.cat([target_pts, mesh_pts], dim=0)
    combined_feat = torch.cat([red_feat, green_feat], dim=0)

    # ONLY HERE do we create a Pointclouds object (for the renderer)
    pc = Pointclouds(points=[combined_pts], features=[combined_feat])

    # Render from multiple angles
    for elev, azim, name in [(30, 30, "diag1"), (30, 120, "diag2"), (60, 0, "diag3")]:
        R, T = look_at_view_transform(dist=camera_distance, elev=elev, azim=azim)
        diag_cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        if pc_renderer is None:
            pc_renderer = PointsRenderer(
                rasterizer=PointsRasterizer(
                    cameras=diag_cameras,
                    raster_settings=RasterizationSettings(
                        image_size=(image_size_y, image_size_x), radius=0.01
                    ),
                ),
                compositor=AlphaCompositor(),
            )

        img = pc_renderer(pc, cameras=diag_cameras)
        save_img_to_temp(img.squeeze().cpu().numpy(), config, f"pc_debug_{name}")


# def get_bounding_box(image: np.ndarray, threshold: float = 0.99) -> Tuple[Tuple[int, int, int, int], int]:
#     """
#     Compute the bounding box for an object in the image based on a simple mask.

#     The image is assumed to be an RGB image. A pixel is considered part of the object if
#     the average value across its RGB channels is less than `threshold`. The image is normalized
#     to [0, 1] if its maximum value is greater than 1.

#     Args:
#         image: A NumPy array of shape (H, W, 3) (or (H, W, 4) if an alpha channel is present).
#         threshold: A float threshold value. Pixels with mean RGB value less than this are "object" pixels.

#     Returns:
#         bbox: A tuple (min_row, min_col, max_row, max_col) defining the bounding box.
#         area: The area of the bounding box (in pixels).
#     """
#     # Normalize the image if needed
#     if image.max() > 1.0:
#         image_norm = image.astype(np.float32) / 255.0
#     else:
#         image_norm = image.copy()

#     # Compute the mask: pixels that are not "almost white" are assumed to be part of the object.
#     mask = (np.mean(image_norm[..., :3], axis=-1) < threshold).astype(np.uint8)

#     # Get non-zero mask indices
#     nonzero = np.nonzero(mask)

#     min_row = int(np.min(nonzero[0]))
#     max_row = int(np.max(nonzero[0]))
#     min_col = int(np.min(nonzero[1]))
#     max_col = int(np.max(nonzero[1]))

#     bbox = (min_row, min_col, max_row, max_col)
#     area = (max_row - min_row) * (max_col - min_col)

#     return bbox, area

# make bounding box from cropped image dimensions, returning bbox and area


def visualize_plane_and_axes(
    plane_to_world_transform: Transform3d,
    renderer: MeshRenderer,
    cameras: FoVPerspectiveCameras,
    output_path: str,
    device: torch.device, # ✅ This is the device we need to use
    object_mesh: Meshes = None,  # ✅ NEW: Optional object mesh to render with plane
    background_image_path: str = None,  # ✅ NEW: Path to input scene image
    image_size: tuple = None,  # ✅ NEW: (width, height) for resizing
):
    """
    Renders a grid representing the plane and colored axes for X, Y, Z
    to help debug the coordinate system and plane orientation.
    Optionally renders an object mesh on the plane.
    Optionally overlays on background image.
    """
    print("Generating plane and axes visualization...")
    
    # Debug: Print transform info
    print(f"Plane-to-world transform matrix:\n{plane_to_world_transform.get_matrix()}")
    
    # 1. Create a large, flat quad mesh for the plane
    # In plane-local coords: Y=0 is the surface, X-Z define the plane
    plane_size = 1.0  # ✅ Made larger to ensure visibility
    plane_verts = torch.tensor([
        [-plane_size, 0, -plane_size],  # (X, Y=0, Z) - on the floor
        [ plane_size, 0, -plane_size],
        [ plane_size, 0,  plane_size],
        [-plane_size, 0,  plane_size],
    ], dtype=torch.float32, device=device) # ✅ Added device
    plane_faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64, device=device) # ✅ Added device
    
    print(f"Plane vertices in local coords (first corner): {plane_verts[0]}")
    transformed_plane_verts = plane_to_world_transform.transform_points(plane_verts.unsqueeze(0)).squeeze(0)
    print(f"Plane vertices after transform (first corner): {transformed_plane_verts[0]}")
    
    # Semi-transparent gray plane
    plane_tex = TexturesVertex(verts_features=torch.ones_like(transformed_plane_verts).unsqueeze(0) * 0.5) 
    plane_mesh = Meshes(verts=[transformed_plane_verts], faces=[plane_faces], textures=plane_tex)

    # 2. Create meshes for the world coordinate axes (trimesh already imported at top)
    axis_len = 1.0
    axis_rad = 0.025
    # X-Axis (Red)
    x_axis_trimesh = trimesh.creation.box(bounds=[(0, -axis_rad, -axis_rad), (axis_len, axis_rad, axis_rad)])
    x_axis_verts = torch.tensor(x_axis_trimesh.vertices, dtype=torch.float32, device=device) # ✅ Added device
    x_axis_faces = torch.tensor(x_axis_trimesh.faces, dtype=torch.int64, device=device) # ✅ Added device
    x_axis_tex_color = torch.tensor([1.0, 0.0, 0.0], device=device) # ✅ Added device
    x_axis_tex = TexturesVertex(verts_features=torch.ones_like(x_axis_verts).unsqueeze(0) * x_axis_tex_color)
    x_axis_mesh = Meshes(verts=[x_axis_verts], faces=[x_axis_faces], textures=x_axis_tex)

    # Y-Axis (Green)
    y_axis_trimesh = trimesh.creation.box(bounds=[(-axis_rad, 0, -axis_rad), (axis_rad, axis_len, axis_rad)])
    y_axis_verts = torch.tensor(y_axis_trimesh.vertices, dtype=torch.float32, device=device) # ✅ Added device
    y_axis_faces = torch.tensor(y_axis_trimesh.faces, dtype=torch.int64, device=device) # ✅ Added device
    y_axis_tex_color = torch.tensor([0.0, 1.0, 0.0], device=device) # ✅ Added device
    y_axis_tex = TexturesVertex(verts_features=torch.ones_like(y_axis_verts).unsqueeze(0) * y_axis_tex_color)
    y_axis_mesh = Meshes(verts=[y_axis_verts], faces=[y_axis_faces], textures=y_axis_tex)

    # Z-Axis (Blue)
    z_axis_trimesh = trimesh.creation.box(bounds=[(-axis_rad, -axis_rad, 0), (axis_rad, axis_rad, axis_len)])
    z_axis_verts = torch.tensor(z_axis_trimesh.vertices, dtype=torch.float32, device=device) # ✅ Added device
    z_axis_faces = torch.tensor(z_axis_trimesh.faces, dtype=torch.int64, device=device) # ✅ Added device
    z_axis_tex_color = torch.tensor([0.0, 0.0, 1.0], device=device) # ✅ Added device
    z_axis_tex = TexturesVertex(verts_features=torch.ones_like(z_axis_verts).unsqueeze(0) * z_axis_tex_color)
    z_axis_mesh = Meshes(verts=[z_axis_verts], faces=[z_axis_faces], textures=z_axis_tex)
    
    # 3. Combine all meshes and render
    from pytorch3d.structures import join_meshes_as_batch
    
    # Create the plane + axes combined mesh
    plane_and_axes = join_meshes_as_batch([plane_mesh, x_axis_mesh, y_axis_mesh, z_axis_mesh])
    
    # ✅ FIX: Render plane/axes and object separately, then composite
    if object_mesh is not None:
        # Render plane+axes first
        image_plane = renderer(meshes_world=plane_and_axes)
        # Render object second
        image_object = renderer(meshes_world=object_mesh)
        
        # Composite: use object's alpha to blend
        alpha_obj = image_object[..., 3:4]  # (1, H, W, 1)
        image_rgb = image_object[..., :3] * alpha_obj + image_plane[..., :3] * (1 - alpha_obj)
        image_np = image_rgb[0].detach().cpu().numpy()
    else:
        # Just render plane+axes
        image = renderer(meshes_world=plane_and_axes)
        image_np = image[0, ..., :3].detach().cpu().numpy()
    
    # ✅ NEW: Composite with background image if provided
    if background_image_path is not None:
        import imageio
        from skimage.transform import resize
        
        # Load background image
        bg_image = imageio.imread(background_image_path)
        
        # Resize background to match renderer output if size provided
        if image_size is not None:
            width, height = image_size
            bg_image = resize(bg_image, (height, width), anti_aliasing=True)
        else:
            # Match rendered image size
            bg_image = resize(bg_image, image_np.shape[:2], anti_aliasing=True)
        
        # Normalize to [0, 1] if needed
        if bg_image.max() > 1:
            bg_image = bg_image.astype(np.float32) / 255.0
        
        # Extract alpha channel from rendered image (if available)
        if image.shape[-1] == 4:
            alpha = image[0, ..., 3].detach().cpu().numpy()[..., None]
        else:
            # Create alpha from non-black pixels
            alpha = (image_np.sum(axis=-1, keepdims=True) > 0.1).astype(np.float32)
        
        # Composite: rendered plane over background
        image_np = image_np * alpha + bg_image[..., :3] * (1 - alpha)
    
    # Save the visualization
    import imageio
    imageio.imwrite(output_path, (image_np * 255).astype(np.uint8))
    print(f"✅ Plane visualization saved to {output_path}")



def clean_mesh(mesh: Meshes) -> Meshes:
    """
    Cleans a PyTorch3D Meshes object by removing invalid vertices and faces,
    and fixes face winding/normal orientation to avoid shading artifacts.

    Returns a new Meshes object with cleaned data.
    """

    verts_list = mesh.verts_list()
    faces_list = mesh.faces_list()
    textures = mesh.textures

    # Create new lists for cleaned data
    new_verts_list = []
    new_faces_list = []

    for i, verts in enumerate(verts_list):
        faces = faces_list[i]
        
        # Check for empty mesh
        if verts.numel() == 0 or faces.numel() == 0:
            # Return an empty tensor of the correct shape if the mesh is empty
            #return torch.empty(0, 3, device=mesh.device)
            raise ValueError(f"Mesh {i} is empty.")
        
        # Check and clean vertices
        valid_verts_mask = torch.isfinite(verts).all(dim=-1)
        if not valid_verts_mask.all():
            num_invalid = (~valid_verts_mask).sum().item()
            print(f"[WARN] Mesh {i} contains {num_invalid} invalid vertices. Cleaning...")
            
            # Find faces that reference invalid vertices
            invalid_faces_mask = ~valid_verts_mask[faces].all(dim=-1)
            
            # Remove invalid faces
            faces = faces[~invalid_faces_mask]

            # Replace invalid vertices with the mean of the valid ones
            # This is a simple fix; for more complex cases, you might remove them
            if valid_verts_mask.any():
                verts[~valid_verts_mask] = verts[valid_verts_mask].mean(dim=0)
            else:
                # All vertices are invalid, can't sample
                return torch.empty(0, 3, device=mesh.device)

        # Ensure consistent winding and sane normals using trimesh (no vertex reindexing)
        try:
            # Work on CPU numpy, keep vertices as-is; only allow face winding/degenerate removal
            tm = trimesh.Trimesh(
                vertices=verts.detach().cpu().numpy(),
                faces=faces.detach().cpu().numpy(),
                process=False,
            )

            # Fix normals/winding and drop degenerate faces
            import trimesh.repair as _repair
            _repair.fix_normals(tm)
            # Remove zero-area/duplicate triangles; do NOT remove unreferenced vertices to preserve indexing
            tm.remove_degenerate_faces()

            # Pull back possibly updated faces (winding may have flipped and some faces removed)
            faces = torch.as_tensor(tm.faces, dtype=faces.dtype, device=faces.device)
        except Exception as e:
            print(f"[WARN] Trimesh normals/winding fix failed on mesh {i}: {e}")

        new_verts_list.append(verts)
        new_faces_list.append(faces)

    # Create a new, clean Meshes object
    clean_mesh = Meshes(verts=new_verts_list, faces=new_faces_list, textures=textures)

    return clean_mesh



def sample_mesh_points(mesh: Meshes, num_samples: int = 10000) -> torch.Tensor:
    """
    Returns a tensor of shape (num_samples, 3) containing points
    uniformly sampled from the surface of `mesh`.
    The points are already in world coordinates because `mesh` is
    defined in world space.
    """
    
    # Now sample from the clean mesh
    pts = sample_points_from_meshes(mesh, num_samples)
    return pts.squeeze(0)


def get_bounding_box(im: np.ndarray) -> Tuple[Tuple[int, int, int, int], int]:
    """
    Get the bounding box of the object in the segmentation image.
    Returns:
        bbox: (x_min, x_max, y_min, y_max)
        area: area of the bounding box
        
    """
    if im is not None:
        im = np.array(im)
        segmentation = np.where(im == 1)

        # Bounding Box
        bbox = 0, 0, 0, 0
        if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))

            bbox = [x_min, x_max, y_min, y_max]

            #########
            # Do what you need to do with the bbox, for example add it to your annotation file
            #########
            return bbox, float((x_max - x_min) * (y_max - y_min))
    else:
        # Handle error case where segmentation image cannot be read or is empty
        print("Error: Segmentation image is empty or cannot be read.")



def normalized_to_camera_space(
    x_norm: float,
    y_norm: float,
    z_depth: float,
    image_w: int,
    image_h: int,
    focal_px: float,
    device
) -> torch.Tensor:
    """
    Unprojects a normalized image coord into camera space using pixel focal.
    """
    cx = image_w / 2.0
    cy = image_h / 2.0

    X = (x_norm * image_w - cx) * z_depth / focal_px
    Y = (y_norm * image_h - cy) * z_depth / focal_px
    Z = z_depth

    return torch.tensor([[X, Y, Z]], dtype=torch.float32, device=device)


def camera_to_world_space(
    cam_coords: torch.Tensor,
    R: torch.Tensor,
    T: torch.Tensor
) -> torch.Tensor:
    """
    Transforms points from camera space into world space.

    Args:
        cam_coords: Tensor of shape (B, 3) or (B, N, 3) giving
                    one or many 3D points in camera coordinates.
        R:          Rotation matrix Tensor of shape (B, 3, 3)
                    that maps camera → world.
        T:          Translation vector Tensor of shape (B, 3)
                    that maps camera → world.

    Returns:
        world_coords: Tensor of same shape as `cam_coords`, but
                    now in world coordinates.
    """
    # Ensure batch dims
    if cam_coords.dim() == 2:
        # (B, 3) @ (B, 3, 3) -> (B, 3)
        world = torch.bmm(cam_coords.unsqueeze(1), R).squeeze(1) + T
    elif cam_coords.dim() == 3:
        # (B, N, 3) @ (B, 3, 3) -> (B, N, 3)
        world = torch.matmul(cam_coords, R.transpose(1, 2)) + T.unsqueeze(1)
    else:
        raise ValueError(
            f"cam_coords must be 2D or 3D, got {cam_coords.dim()}D")

    return world


# def regularize_depth_map(depth, mask, image_ref, iterations=3, kernel_size=3):
#     """Reduces noise while preserving edges in depth map"""
#     # Convert to 8-bit for OpenCV (preserves relative depth)
#     depth_8bit = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
#     # Apply joint bilateral filter using the RGB image as guide
#     # This preserves depth edges that align with image edges
#     rgb_guide = cv2.resize(image_ref, (depth.shape[1], depth.shape[0]))
#     rgb_guide = (rgb_guide * 255).astype(np.uint8)
    
#     depth_regularized = cv2.ximgproc.jointBilateralFilter(
#         depth_8bit, rgb_guide, 
#         d=15, sigmaColor=75, sigmaSpace=75
#     ).astype(np.float32) / 255.0
    
#     # Re-normalize to [0,1] within the object mask
#     depth_masked = depth_regularized[mask > 0.25]
#     if len(depth_masked) > 0:
#         depth_regularized = (depth_regularized - depth_masked.min()) / (depth_masked.max() - depth_masked.min() + 1e-6)
    
#     # Apply slight morphological closing to remove small holes
#     kernel = np.ones((kernel_size, kernel_size), np.uint8)
#     depth_regularized = cv2.morphologyEx(
#         (depth_regularized * 255).astype(np.uint8), 
#         cv2.MORPH_CLOSE, kernel, iterations=iterations
#     ).astype(np.float32) / 255.0

#     # [x,x,3] depth regularized to [x,x]
#     depth_regularized = depth_regularized[..., 0]

#     return depth_regularized * mask  # Re-apply mask

# def regularize_depth_map(depth, mask, image_ref, sigma_spatial=3.0, sigma_color=0.1):
#     """
#     Proper depth regularization that preserves 3D structure
    
#     Args:
#         depth: Float32 depth map [0,1]
#         mask: Binary object mask [0,1]
#         image_ref: RGB image [0,1] for edge guidance
#         sigma_spatial: Filter window size (pixels)
#         sigma_color: Color similarity threshold (0-1)
#     """
#     # Apply mask first (critical!)
#     depth = depth * mask
    
#     # Convert to consistent format
#     depth_32 = depth.astype(np.float32)

#     # Ensure the RGB guide matches the depth dimensions
#     H, W = depth_32.shape[:2]
#     if image_ref is None:
#         # fallback: use gray image created from depth (no guidance)
#         rgb_guide = cv2.cvtColor((depth_32 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#     else:
#         rgb_uint8 = (image_ref * 255).astype(np.uint8)
#         if rgb_uint8.shape[0] != H or rgb_uint8.shape[1] != W:
#             rgb_guide = cv2.resize(rgb_uint8, (W, H), interpolation=cv2.INTER_LINEAR)
#         else:
#             rgb_guide = rgb_uint8

#     # Apply JOINT BILATERAL FILTER with reasonable parameters
#     # These values preserve depth edges that align with RGB edges but don't over-smooth
#     try:
#         depth_regularized = cv2.ximgproc.jointBilateralFilter(
#             depth_32,
#             rgb_guide,
#             d=max(3, int(sigma_spatial)),        # Window size
#             sigmaColor=max(1.0, sigma_color * 255.0),
#             sigmaSpace=max(1.0, sigma_spatial)
#         )
#     except Exception:
#         # Fallback: use single-image bilateral filter on the depth map
#         depth_regularized = cv2.bilateralFilter(
#             (depth_32 * 255).astype(np.uint8),
#             d=max(3, int(sigma_spatial)),
#             sigmaColor=max(1.0, sigma_color * 255.0),
#             sigmaSpace=max(1.0, sigma_spatial)
#         ).astype(np.float32) / 255.0
    
#     #  CRITICAL: DO NOT re-normalize the entire depth map!
#     # Preserve the original depth relationships
    
#     # Optional: Very gentle hole filling (only where mask exists)
#     mask_uint8 = (mask * 255).astype(np.uint8)
#     depth_filled = cv2.inpaint(
#         (depth_regularized * 255).astype(np.uint8),
#         255 - mask_uint8,
#         inpaintRadius=3,
#         flags=cv2.INPAINT_NS
#     ).astype(np.float32) / 255.0
    
#     return depth_filled * mask  # Re-apply mask




def dice_loss(P, T, epsilon=1e-6):
    # Flatten tensors (supports 2D/3D masks)
    P_flat = P
    T_flat = T

    intersection = torch.sum(P_flat * T_flat)
    denominator = torch.sum(P_flat) + torch.sum(T_flat)

    dice = (2. * intersection + epsilon) / (denominator + epsilon)
    return 1 - dice


def extract_camera_from_json(image_path: str, output_folder: str) -> str:
    """
    Extracts camera data for a given image from the 3D-Front meta.json file,
    and saves it in Dust3r-style npz format.

    Parameters:
        image_path (str): Full path to the image (e.g. '../3D-FRONT-RENDER/.../render_0005.webp')
        output_folder (str): Folder where the camera .npz should be saved

    Returns:
        str: Path to the saved .npz file
    """
    image_filename = os.path.basename(image_path)  # e.g. 'render_0005.webp'
    image_index = os.path.splitext(image_filename)[0].split('_')[-1]  # '0005'

    # Assume meta.json is in the same folder
    meta_json_path = os.path.join(os.path.dirname(image_path), 'meta.json')
    if not os.path.exists(meta_json_path):
        raise FileNotFoundError(f"meta.json not found at: {meta_json_path}")

    with open(meta_json_path, 'r') as f:
        meta = json.load(f)

    # Find location with matching index
    matched_location = None
    for loc in meta.get("locations", []):
        if loc.get("index") == image_index:
            matched_location = loc
            break

    if matched_location is None:
        raise ValueError(
            f"No location with index '{image_index}' found in metadata.")

    # Grab transform matrix and any render frame for dimensions
    transform = np.array(matched_location["transform_matrix"])  # 4x4
    extrinsic = transform[:4, :4]  # 3x4

    # For position string
    position = matched_location["position"]

    # Find a frame (assume all types share size)
    frame = next(
        (f for f in matched_location["frames"] if f["type"] == "render"), None)
    if frame is None:
        raise ValueError(f"No 'render' frame found for index {image_index}")

    width = frame["width"]
    height = frame["height"]

    # Focal length in pixels
    camera_angle_x = meta["camera_angle_x"]
    focal_px = 0.5 * width / np.tan(0.5 * camera_angle_x)

    # Optional additional fields
    sensor_width = meta.get("sensor_width", 36.0)  # mm
    camera_lens = meta.get("camera_lens", 50.0)    # mm

    # Prepare and save npz
    camera_data = {
        "extrinsic": extrinsic.astype(np.float32),
        "focal": np.float32(focal_px),
        "image_size": np.array([width, height], dtype=np.int32),
        "sensor_width": np.float32(sensor_width),
        "camera_lens": np.float32(camera_lens),
        "camera_angle_x": np.float32(camera_angle_x),
        "position": position.encode('utf-8')
    }

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "camera_dataset.npz")
    np.savez(output_path, **camera_data)

    return output_path


