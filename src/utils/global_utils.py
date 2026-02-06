from math import floor
import os
import json
from typing import Tuple

import numpy as np
import shutil
import yaml


import logging

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import PIL


def create_segmentation_layout(
    original_image_path, 
    extracted_image_path, 
    output_path="output_layout.png",
    target_width=1280,                # 1. Resize input to this width
    panel_bg_color=(230, 230, 230),   # Light gray background for right side
    card_bg_color=(255, 255, 255),    # White background for the object card
    border_color=(0, 0, 0),           # Black border
    border_width=5,
    corner_radius=20,
    text_label="Extracted Object"
):
    """
    Creates a side-by-side layout with a standardized width and a square object container.
    """
    
    # --- 1. Load and Resize Images ---
    try:
        img_original = Image.open(original_image_path).convert("RGBA")
        img_extracted = Image.open(extracted_image_path).convert("RGBA")
    except FileNotFoundError as e:
        print(f"Error loading images: {e}")
        return

    # Calculate new height to maintain aspect ratio for the original image
    aspect_ratio = img_original.height / img_original.width
    new_height = int(target_width * aspect_ratio)
    
    # Resize original image (high quality)
    img_original = img_original.resize((target_width, new_height), Image.Resampling.LANCZOS)
    
    # Dimensions for the layout
    base_width = target_width
    base_height = new_height
    
    # CHANGE: The right panel is now a SQUARE (width matches height)
    panel_width = base_height  
    
    # Create the base canvas
    total_width = base_width + panel_width
    canvas = Image.new("RGBA", (total_width, base_height), (255, 255, 255, 255))
    
    # --- 2. Draw Layout ---
    
    # Paste Original Image on the Left
    canvas.paste(img_original, (0, 0))
    
    # Draw the Right Panel Background
    draw = ImageDraw.Draw(canvas)
    panel_rect = [base_width, 0, total_width, base_height]
    draw.rectangle(panel_rect, fill=panel_bg_color)
    
    # --- 3. Create the Square Card ---
    
    # Define constraints
    # CHANGE: Increased bottom space for the larger font
    bottom_space_for_text = int(base_height * 0.08) 
    margin = int(panel_width * 0.04)     # 7.5% margin on sides
    top_margin = int(base_height * 0.02)
    
    # Calculate maximum possible square size
    # It must fit within the width (minus margins) AND height (minus text space)
    max_w = panel_width - (margin * 2)
    max_h = base_height - bottom_space_for_text - top_margin
    
    square_size = min(max_w, max_h)
    
    # Calculate coordinates to center the square in the panel
    # Center horizontally in the panel
    panel_center_x = base_width + (panel_width // 2)
    card_x1 = panel_center_x - (square_size // 2)
    card_x2 = card_x1 + square_size
    
    # Center vertically in the available space above the text area
    vertical_space = base_height - bottom_space_for_text
    card_y1 = (vertical_space - square_size) // 2
    card_y2 = card_y1 + square_size

    # Draw Rounded Rectangle (The Card)
    draw.rounded_rectangle(
        (card_x1, card_y1, card_x2, card_y2), 
        radius=corner_radius, 
        fill=card_bg_color, 
        outline=border_color, 
        width=border_width
    )
    
    # --- 4. Fit and Paste Extracted Object ---
    
    # Resize extracted object to fit inside the square with padding
    # CHANGE: Increased padding to inset the object more
    padding = 40
    target_obj_size = square_size - (padding * 2)
    
    # Calculate aspect ratio of the object
    obj_ratio = img_extracted.width / img_extracted.height
    
    if obj_ratio > 1:
        # Wider than tall
        new_obj_w = target_obj_size
        new_obj_h = int(new_obj_w / obj_ratio)
    else:
        # Taller than wide
        new_obj_h = target_obj_size
        new_obj_w = int(new_obj_h * obj_ratio)
        
    img_extracted_resized = img_extracted.resize((new_obj_w, new_obj_h), Image.Resampling.LANCZOS)
    
    # Center the object in the card
    paste_x = card_x1 + (square_size - new_obj_w) // 2
    paste_y = card_y1 + (square_size - new_obj_h) // 2
    
    canvas.paste(img_extracted_resized, (paste_x, paste_y), img_extracted_resized)
    
    # --- 5. Add Text ---
    
        # CHANGE: Significantly increased font size logic
        # 12% of height or at least 60px
    font_size = max(int(base_height * 0.035), 55) 
    


    fonts_dir = os.path.join("../", "fonts")
    
    possible_fonts = [
        # Project fonts (create a fonts/ directory in your project root)
        os.path.join(fonts_dir, "Roboto-Bold.ttf"),
        os.path.join(fonts_dir, "Arial-Bold.ttf"),
        # System fonts (Linux)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        # System fonts (Windows)
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/Arial-BoldMT.ttf",
        # System fonts (Mac)
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    
    font = None
    for f in possible_fonts:
        try:
            if os.path.exists(f):
                font = ImageFont.truetype(f, font_size)
                print(f"✓ Using font: {os.path.basename(f)}")
                break
        except (OSError, IOError) as e:
            continue
    
    if font is None:
        print("⚠ No TrueType font found, using PIL default font (text may be small).")
        font = ImageFont.load_default()


    # Calculate text position
    left, top, right, bottom = draw.textbbox((0, 0), text_label, font=font)
    text_w = right - left
    text_h = bottom - top
    
    # Center text horizontally in panel
    text_x = base_width + (panel_width - text_w) // 2
    
    # Place text in the space below the card
    text_area_start = card_y2
    text_area_height = base_height - card_y2
    text_y = text_area_start + (text_area_height - text_h) // 2
    
    draw.text((text_x, text_y), text_label, fill=(0, 0, 0), font=font)
    
    # Save
    canvas.save(output_path)
    print(f"Layout saved to {output_path}")


def extract_AQ_object(img:PIL.Image = None, target_width: int = 1280) -> PIL.Image:
    """
    Extracts the AQ object from the right side of the image.:
    Args:
        image (PIL.Image): The input image containing the AQ UI.
    Returns:
        PIL.Image: The cropped image containing only the AQ object.
    """



    # We know the left side (original image) was forced to target_width (1280).
    base_width = target_width
    
    # In your logic, the overall image height determined the right panel size.
    # So we just read the height from the loaded image.
    base_height = img.height
    
    # Verify image integrity (Optional sanity check)
    # The generation script set: panel_width = base_height
    # So total width should be roughly base_width + base_height.
    # (We skip strict assertion in case of minor rounding errors, but it's good to know)
    
    panel_width = base_height
    
    # --- 2. Recalculate the Box Coordinates ---
    # These formulas are identical to your generation script.
    
    bottom_space_for_text = int(base_height * 0.08) 
    margin = int(panel_width * 0.04)
    top_margin = int(base_height * 0.02)
    
    # Calculate maximum possible square size
    max_w = panel_width - (margin * 2)
    max_h = base_height - bottom_space_for_text - top_margin
    
    square_size = min(max_w, max_h)
    
    # Calculate Center X
    panel_center_x = base_width + (panel_width // 2)
    card_x1 = panel_center_x - (square_size // 2)
    card_x2 = card_x1 + square_size
    
    # Calculate Center Y
    vertical_space = base_height - bottom_space_for_text
    card_y1 = (vertical_space - square_size) // 2
    card_y2 = card_y1 + square_size
    
    # The border_width in generation was 5. 
    # Do you want to include the black border or crop inside it?
    # Usually, for extraction, you want to crop INSIDE the border to get just the object.
    # Set this to 0 if you want to keep the black border.
    border_offset = 5 
    
    crop_box = (
        card_x1 + border_offset, 
        card_y1 + border_offset, 
        card_x2 - border_offset, 
        card_y2 - border_offset
    )
    
    # --- 3. Crop and Save ---
    extracted_box = img.crop(crop_box)

    return extracted_box





def match_pointclouds(src_pc, target_pc):
    """
    Approximates the transformation (uniform scale, rotation, translation)
    to align a source point cloud to a target point cloud using PCA.

    This method provides a coarse, one-shot alignment.

    Args:
        src_pc (np.ndarray or torch.Tensor): The source point cloud, shape (N, 3) or (1, N, 3).
        target_pc (np.ndarray or torch.Tensor): The target point cloud, shape (M, 3) or (1, M, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]:
        - np.ndarray: The transformed source point cloud, aligned with the target, shape (N, 3).
        - np.ndarray: The 4x4 homogeneous affine transformation matrix that maps
                      the original source cloud to the aligned position.
    """
    import torch

    # Handle PyTorch GPU tensors by moving them to CPU
    if isinstance(src_pc, torch.Tensor):
        src_pc = src_pc.detach().cpu().numpy()
    else:
        src_pc = np.asarray(src_pc)

    if isinstance(target_pc, torch.Tensor):
        target_pc = target_pc.detach().cpu().numpy()
    else:
        target_pc = np.asarray(target_pc)

    # --- FIX STARTS HERE ---
    # Ensure point clouds are 2D (N, 3) by squeezing a potential batch dimension
    if src_pc.ndim == 3:
        # If shape is (1, N, 3), squeeze it to (N, 3)
        if src_pc.shape[0] == 1:
            src_pc = src_pc.squeeze(0)
        else:
            raise ValueError(
                f"Source point cloud has batch size {src_pc.shape[0]}, expected a single cloud."
            )

    if target_pc.ndim == 3:
        if target_pc.shape[0] == 1:
            target_pc = target_pc.squeeze(0)
        else:
            raise ValueError(
                f"Target point cloud has batch size {target_pc.shape[0]}, expected a single cloud."
            )
    # --- FIX ENDS HERE ---

    # 1. Center the point clouds by subtracting their centroids
    src_centroid = np.mean(src_pc, axis=0)
    target_centroid = np.mean(target_pc, axis=0)
    src_centered = src_pc - src_centroid
    target_centered = target_pc - target_centroid

    # 2. Compute a uniform scale factor
    src_extent = np.linalg.norm(
        np.max(src_centered, axis=0) - np.min(src_centered, axis=0)
    )
    target_extent = np.linalg.norm(
        np.max(target_centered, axis=0) - np.min(target_centered, axis=0)
    )

    if src_extent == 0:
        scale = 1.0
    else:
        scale = target_extent / src_extent

    src_scaled = src_centered * scale

    # 3. Compute the rotation matrix using PCA
    cov_src = np.cov(src_scaled, rowvar=False)
    cov_target = np.cov(target_centered, rowvar=False)
    _, src_axes = np.linalg.eigh(cov_src)
    _, target_axes = np.linalg.eigh(cov_target)
    R = target_axes @ src_axes.T

    # 4. Handle potential reflections
    if np.linalg.det(R) < 0:
        target_axes[:, 0] *= -1
        R = target_axes @ src_axes.T

    # 5. Apply the full transformation
    transformed_src_pc = (R @ (src_centered * scale).T).T + target_centroid

    # 6. Construct the equivalent 4x4 affine transformation matrix
    T = np.identity(4)
    T[:3, :3] = R * scale
    T[:3, 3] = target_centroid - (R * scale) @ src_centroid

    return transformed_src_pc, T


def depth_from_image(
    image_path, config: dict = None, large: bool = False
) -> np.ndarray:
    """
    Uses the Depth Anything model to estimate depth from an image.
    Use large for Marigold else Depth Anything V2 small

    """
    from transformers import pipeline
    from PIL import Image
    import torch
    import diffusers

    depth_path = config["depth_scene"]
    print(f"Absolute Depth Path: {os.path.abspath(depth_path)}")

    if large:
        pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
            "prs-eth/marigold-depth-v1-1", variant="fp16", torch_dtype=torch.float16
        ).to("cuda")

        image = diffusers.utils.load_image(image_path)

        depth_scene = pipe(image)

        print("Shape prediction: ", depth_scene.prediction.shape)

        depth_16bit = pipe.image_processor.export_depth_to_16bit_png(
            depth_scene.prediction
        )
        # divide by 256 to get range 0-255
        print(
            "Shape depth_16bit: ", depth_16bit[0].size, " Mode: ", depth_16bit[0].mode
        )
        depth_array = np.array(depth_16bit[0])  # Convert PIL Image to numpy array
        depth_normalized = depth_array // 256  # Integer division to get range 0-255
        depth_16bit_image = Image.fromarray(depth_normalized.astype(np.uint8))

        depth_16bit_image.save(depth_path)

        print("Saved depth map to: ", os.path.abspath(depth_path))
        depth_scene = np.array(depth_16bit_image)

    else:
        """
        Uses the Depth Anything model to estimate depth from an image.
        """
        pipe = pipeline(
            task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf"
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


def save_img_to_temp(image, config, name):
    """
    Saves an image to the temporary directory specified in the config.
    """
    import imageio
    from skimage import img_as_ubyte

    temp_dir = config.get("temp", "temp")
    os.makedirs(temp_dir, exist_ok=True)

    name = name + ".png" if not name.endswith(".png") else name

    # clamp image values to [0, 255]
    image = np.clip(image, 0, 255)
    # clamp to -1 and 1 if float
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = np.clip(image, -1, 1)

    path = os.path.join(temp_dir, name)
    imageio.imwrite(path, img_as_ubyte(image))


def clear_output_directory(output_dir):
    """
    Clears the contents of the specified output directory.
    Parameters:
        output_dir (str): Path to the output directory to be cleared.
    """

    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    # Remove all contents of the output directory
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)

    logging.debug(f"Cleared contents of output directory: {output_dir}")


def load_config(path):
    """
    Loads a YAML configuration file.
    Parameters:
        path (str): Path to the YAML configuration file.
    Returns:
        dict: Parsed configuration as a dictionary.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes."""
    # box format: [x_min, y_min, x_max, y_max]

    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # If there's no intersection, return 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


# create complete glb scene from all glb files in a directory
def create_glb_scene(
    input_dir,
    output_path="output/glb/combined_scene.glb",
    config: dict = None,
    device="cuda",
) -> str:
    """
    Combines all .glb files in a directory into a single GLB scene.

    Args:
        input_dir (str): Directory containing individual .glb files.
        output_dir (str): Directory where the combined .glb will be saved.
        output_name (str): Filename for the combined output GLB.
        device (str): Currently unused; reserved for future GPU-based processing.
    """

    import trimesh

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_scene = trimesh.Scene()

    list_aluminium = config.get("list_aluminium_scene", [])
    # Use a set for O(1) membership checks; normalize by removing extensions and lowercasing
    aluminium_names = {os.path.splitext(name)[0].lower() for name in list_aluminium}

    for filename in sorted(os.listdir(input_dir)):
        # Skip non-GLB files and files with scene in its name
        if not filename.endswith(".glb") or "scene" in filename.lower():
            continue

        glb_path = os.path.join(input_dir, filename)
        model_name = os.path.splitext(filename)[0]
        model_key = model_name.lower()

        try:
            loaded = trimesh.load_scene(
                glb_path, force="scene"
            )  # Always return a Scene
            if not isinstance(loaded, trimesh.Scene):
                temp_scene = trimesh.Scene()
                temp_scene.add_geometry(loaded)
                loaded = temp_scene

            for name, geom in loaded.geometry.items():
                # Check if the geometry has a PBR material and modify it
                if hasattr(geom, "visual") and hasattr(geom.visual, "material"):
                    logging.debug(f"Modifying material for {name} in {model_name}")
                    if isinstance(
                        geom.visual.material, trimesh.visual.material.PBRMaterial
                    ):
                        logging.debug(
                            f"Original metallic: {geom.visual.material.metallicFactor}"
                        )
                        # Set the metallic factor to 0.05
                        geom.visual.material.metallicFactor = config.get(
                            "metallic", 0.1
                        )
                        geom.visual.material.roughnessFactor = config.get(
                            "roughness", 0.5
                        )
                        logging.info(
                            f"Updated metallic: {geom.visual.material.metallicFactor}"
                        )

                        # Special-case overrides based on lists from config (hash-set membership)
                        if model_key in aluminium_names:
                            mat = geom.visual.material
                            mat.metallicFactor = config.get(
                                "metallic_aluminium", 0.95
                            )
                            mat.roughnessFactor = config.get(
                                "roughness_aluminium", 0.1
                            )
                            if hasattr(mat, "baseColorTexture"):
                                mat.baseColorTexture = None
                            
                            mat.baseColorFactor = config.get(
                                "albedo_aluminium", [0.1, 0.1, 0.1, 1.0]
                            )
                            logging.info(
                                f"Aluminium override - metallic: {mat.metallicFactor}, roughness: {mat.roughnessFactor}, albedo: {mat.baseColorFactor}"
                            )

                            
                combined_scene.add_geometry(geom, node_name=f"{model_name}_{name}")

        except Exception as e:
            logging.error(f"[ERROR] Failed to load {glb_path}: {e}")

    try:
        combined_scene.export(output_path, file_type="glb")
        logging.info(f"[INFO] Combined GLB saved to: {output_path}")
    except Exception as e:
        logging.error(f"[ERROR] Failed to export combined scene: {e}")

    return output_path


# create a point cloud scene .ply from object ply files in a directory
def create_pred_ply_scene(
    input_dir, output_path="output/pointclouds/combined_scene.ply", device="cuda"
) -> str:
    """
    Combines all .ply files in a directory into a single PLY point cloud.

    Args:
        input_dir (str): Directory containing individual .ply files.
        output_dir (str): Directory where the combined .ply will be saved.
        output_name (str): Filename for the combined output PLY.
        device (str): Currently unused; reserved for future GPU-based processing.
    """

    import trimesh

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for filename in sorted(os.listdir(input_dir)):
        # Skip non-GLB files and files with scene in its name
        if not filename.endswith(".ply") or "scene" in filename.lower():
            continue

        ply_path = os.path.join(input_dir, filename)
        model_name = os.path.splitext(filename)[0]

        try:
            # load points
            loaded = trimesh.load(ply_path, file_type="ply")

            if not isinstance(loaded, trimesh.PointCloud):
                print(f"[ERROR] Loaded geometry is not a point cloud: {ply_path}")
                continue

            # add points to all other points
            if "point_cloud" not in locals():
                point_cloud = loaded
            else:
                point_cloud += loaded

        except Exception as e:
            logging.error(f"Failed to load {ply_path}: {e}")

    print(
        "PointCloud : ",
        point_cloud,
        " Type: ",
        type(point_cloud),
        " and Shape : ",
        point_cloud.shape,
    )

    # point_cloud = trimesh.points.PointCloud(point_cloud)

    try:
        point_cloud.export(output_path, file_type="ply")
        logging.info(f"Combined PLY saved to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to export combined scene: {e}")

    return output_path


def save_point_cloud(points, output_path: str, blender_readable=False):
    """
    Save a point cloud to a PLY file.
    Apply transformations to make it Blender-readable.

    Args:
        points (torch tensor): Point cloud data of shape (N, 3).
        output_path (str): Path to save the PLY file.
    """
    import trimesh
    import torch

    # check if torch tensor or numpy array
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy().squeeze()
    elif not isinstance(points, np.ndarray):
        raise TypeError("Points must be a torch tensor or numpy array.")

    if blender_readable:
        points[:, 2] *= -1  # Flip new Z axis (original Y)
        points = points[:, [0, 2, 1]]  # Swap Y and Z

    point_cloud = trimesh.points.PointCloud(points)
    point_cloud.export(output_path)
    logging.debug(f"Point cloud saved to {output_path}")

    return points


def load_glb_to_point_cloud(
    glb_path,
    output_file=None,
    num_samples=20480,
    skip_textures_material: bool = False,
    device="cuda",
    save=True,
):
    """
    Load a GLB file, convert it to a PyTorch3D mesh, and sample a point cloud.

    Args:
        glb_path (str): Path to the GLB file.
        num_samples (int): Number of points to sample.
        device (str): Device for PyTorch tensors (e.g., 'cuda' or 'cpu').

    Returns:
        Tensor: Sampled point cloud of shape (1, num_samples, 3)
    """
    import trimesh
    from pytorch3d.structures import Meshes
    from pytorch3d.ops import sample_points_from_meshes
    import torch

    # Load with trimesh (skip textures since we only need geometry for point clouds)
    mesh_or_scene = trimesh.load(
        glb_path,
        force="scene",
        skip_texture=skip_textures_material,
        skip_materials=skip_textures_material,
    )

    # Collect all geometry meshes into a single Trimesh
    if isinstance(mesh_or_scene, trimesh.Scene):
        mesh = mesh_or_scene.to_geometry()  # one mesh
    else:
        mesh = mesh_or_scene

    # Ensure we have a triangular mesh
    if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.shape[1] != 3:
        raise ValueError(f"Invalid or non-triangular mesh in: {glb_path}")

    # Convert to PyTorch3D
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)
    mesh_p3d = Meshes(verts=[vertices], faces=[faces])

    # Sample points
    points = sample_points_from_meshes(mesh_p3d, num_samples=num_samples)

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Create and save point cloud
    if save:
        points = save_point_cloud(points, output_file, blender_readable=False)

    return torch.from_numpy(points).unsqueeze(0).to(device)


def apply_icp_results_to_glb(scene_path, R, t, output_path=None) -> str:
    """
    Apply ICP results to GLB scene with proper coordinate system handling.
    Converts from Y-up (PyTorch3D) to Z-up (GLTF/Blender) coordinate system.
    """
    import trimesh

    # Convert to numpy and ensure proper shape
    R_np = R.cpu().numpy().squeeze().astype(np.float64)
    t_np = t.cpu().numpy().squeeze().astype(np.float64)

    # Create Y-up to Z-up conversion matrix (90 degree rotation around X-axis)
    # This converts: Y-up (PyTorch3D) -> Z-up (GLTF/Blender)
    y_to_z_conversion = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)

    # Convert the rotation matrix
    # R_np = y_to_z_conversion @ R_np @ y_to_z_conversion.T

    # Convert the translation vector
    # t_np = y_to_z_conversion @ t_np

    # Create 4x4 transformation matrix with converted values
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R_np
    transform[:3, 3] = t_np

    # Load scene
    scene = trimesh.load(scene_path, force="scene")

    # PROPER WAY TO HANDLE TRIMESH SCENE GRAPH:
    try:
        # Get the base frame (root node)
        base_frame = scene.graph.base_frame

        # Get current transform of the base frame
        current_transform, _ = scene.graph.get_transform(base_frame)

        # Compose the new transform (ICP transform applied AFTER existing transform)
        new_transform = transform @ current_transform

        # Set the new transform for the base frame
        scene.graph.set_transform(base_frame, new_transform)
        print("Successfully applied transform through scene graph")
    except Exception as e:
        print(f"Scene graph method failed: {e}")
        print("Falling back to direct geometry transformation")
        # Fallback: Apply transform directly to all geometries
        for name, geometry in scene.geometry.items():
            geometry.apply_transform(transform)

    # Export the transformed scene
    if output_path is None:
        output_path = scene_path.replace(".glb", "_icp.glb")
        print(f"Saving ICP transformed GLB to: {output_path}")

    scene.export(output_path)
    print(f"Transformed scene saved to: {output_path}")
    return output_path


## Matrix conversions between Blender and PyTorch3D


def P2B(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Convert a PyTorch3D rotation and translation to a Blender 4x4 matrix.
    """
    P2B_R1 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    P2B_R2 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
    P2B_T = np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    vec4w = np.array([[0, 0, 0, 1]], dtype=np.float64)
    Bcol3 = P2B_T @ R @ T
    B3x3 = P2B_R1 @ R @ P2B_R2
    B3x4 = np.concatenate([B3x3, Bcol3[:, None]], axis=1)
    B = np.concatenate([B3x4, vec4w], axis=0)
    return B


# blender to pytorch3d
def B2P(B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a Blender 4x4 matrix to a PyTorch3D rotation and translation.
    """
    B2P_R1 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    B2P_R2 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
    B2P_T = np.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    R = B2P_R1 @ B[:3, :3] @ B2P_R2
    T = B2P_T @ B[:3, 3] @ R
    return R, T
