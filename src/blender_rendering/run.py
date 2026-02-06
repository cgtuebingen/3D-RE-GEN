import bpy
import os
import numpy as np
import mathutils
import yaml
#import tempfile
import argparse

print(os.getcwd())
# path append ../src/
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, ".."))
if src_dir not in sys.path:
    sys.path.append(src_dir)
    print(f"Added {src_dir} to sys.path")

from utils.global_utils import load_config
from PIL import Image
import subprocess, sys
import zipfile
import math

import shutil


# # --- Configuration ---
# # NOTE: Using Blender 4.2.3 (from your logs), but keeping 4.0 as per your script variable.
# # If you are using Blender 4.2.3, you should use '4.2' or '4.3' in the path instead of '4.0'.
# BLENDER_VERSION = "4.2" # Change this to "4.2" or "4.3" if your Blender version is newer
# ADDON_NAME = "io_mesh_ply"
# # ✅ CORRECTED URL: This points to the main Blender repo, which contains the built-in addons.
# ADDON_URL = "https://projects.blender.org/blender/blender/archive/main.zip" 
# #"https://projects.blender.org/blender/blender-addons/archive/main.zip" #"https://projects.blender.org/blender/blender/archive/main.zip"
# # ---------------------

# addon_dir = os.path.expanduser(f"../blender/{BLENDER_VERSION}/scripts/addons/{ADDON_NAME}")
# addon_zip = os.path.expanduser(f"~/tmp/{ADDON_NAME}_blender_main.zip") # Changed zip name to avoid confusion

# # 1. ATTEMPT TO READ THE ADDON DIRECTORY
# try:
#     # Check if the directory exists and is non-empty
#     addon_dir_files = os.listdir(addon_dir)
#     addon_missing = not addon_dir_files
    
# except FileNotFoundError:
#     # The directory does not exist, so the addon is missing
#     addon_missing = True

# # 2. INSTALL ADDON IF MISSING
# if addon_missing:
#     print(f"⚠️ {ADDON_NAME} addon files missing, fetching from main Blender repository...")

#     # a. Ensure the temporary directory for the zip file exists
#     os.makedirs(os.path.dirname(addon_zip), exist_ok=True)


#     subprocess.check_call([
#         "wget",
#         "-O", addon_zip,
#         ADDON_URL])

    
#     # c. Ensure the final addon destination directory exists
#     if os.path.exists(addon_dir):
#         shutil.rmtree(addon_dir)
#     os.makedirs(addon_dir, exist_ok=True)

#     # d. Unpack ONLY the specific addon folder from the zip into addon_dir
#     with zipfile.ZipFile(addon_zip, 'r') as zip_ref:
        
#         # ✅ FINAL CORRECTED LOGIC: The path does not contain 'release'.
#         # It should be like: 'blender-main/scripts/addons/io_mesh_ply/'
#         addon_source_path = None
#         try:
#             # This is the correct path segment to search for.
#             expected_path_segment = f'/scripts/addons/{ADDON_NAME}/'
#             addon_source_path = next(name for name in zip_ref.namelist() if expected_path_segment in name and name.endswith('/'))
#         except StopIteration:
#             print(f"❌ ERROR: Could not find '{expected_path_segment}' inside the downloaded zip file.")
#             print("   This is unexpected. The repository structure may have changed.")
#             exit(1)

#         # The rest of your extraction logic should now work perfectly
#         # as it will be given the correct source path.
#         print(f"Found addon at: {addon_source_path}")
#         print("Extracting files...")
#         for file_in_zip in zip_ref.namelist():
#             if file_in_zip.startswith(addon_source_path):
#                 relative_path = os.path.relpath(file_in_zip, addon_source_path)
#                 dest_path = os.path.join(addon_dir, relative_path)
                
#                 if file_in_zip.endswith('/') and not os.path.exists(dest_path):
#                     os.makedirs(dest_path)
#                 elif not file_in_zip.endswith('/'):
#                     if not os.path.exists(os.path.dirname(dest_path)):
#                         os.makedirs(os.path.dirname(dest_path))
#                     with zip_ref.open(file_in_zip) as source, open(dest_path, "wb") as target:
#                         shutil.copyfileobj(source, target)
                            
#         print(f"✅ {ADDON_NAME} installed successfully at {addon_dir}")






def set_pc_for_render(pc_obj, scale=0.005):
    # Create a default material
    mat = bpy.data.materials.new(name="PointCloudMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    principled_bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    output = nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(principled_bsdf.outputs['BSDF'], output.inputs['Surface'])
    # set base color to green
    principled_bsdf.inputs['Base Color'].default_value = (0.75, 0.1, 0.1, 0.8)  # RGBA
    pc_obj.data.materials.append(mat)
    

    # Add Geometry Nodes modifier
    geo_mod = pc_obj.modifiers.new(name="PointCloudGeo", type='NODES')
    geo_mod.show_viewport = True
    geo_mod.show_render = True
    if not geo_mod.node_group:
        geo_mod.node_group = bpy.data.node_groups.new("PointCloudGeoTree", 'GeometryNodeTree')
    ng = geo_mod.node_group
    ng.nodes.clear()

    # Add input/output to interface
    ng.interface.new_socket(name="Geometry", in_out='INPUT', socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type="NodeSocketGeometry")

    # Create nodes
    input_node = ng.nodes.new("NodeGroupInput")
    output_node = ng.nodes.new("NodeGroupOutput")
    mesh_to_points = ng.nodes.new("GeometryNodeMeshToPoints")
    mesh_to_points.inputs['Radius'].default_value = scale  # Point size
    set_material = ng.nodes.new("GeometryNodeSetMaterial")
    set_material.inputs['Material'].default_value = mat

    # Position nodes
    input_node.location = (-400, 0)
    mesh_to_points.location = (-200, 0)
    set_material.location = (0, 0)
    output_node.location = (200, 0)

    # Connect nodes
    ng.links.new(input_node.outputs['Geometry'], mesh_to_points.inputs['Mesh'])
    ng.links.new(mesh_to_points.outputs['Points'], set_material.inputs['Geometry'])
    ng.links.new(set_material.outputs['Geometry'], output_node.inputs['Geometry'])




def ply_work_around(ply_path: str = "output/pointclouds/scene/combined_scene.ply", obj_name="ImportedPLY"):
    vertices = []
    faces = []
    header_ended = False
    num_vertices = 0
    num_faces = 0
    vertex_section = False
    face_section = False
    
    with open(ply_path, "r") as f:
        lines = f.readlines()
    
    # parse header
    header = []
    i = 0
    for line in lines:
        header.append(line.strip())
        if line.startswith("element vertex"):
            num_vertices = int(line.split()[-1])
        if line.startswith("element face"):
            num_faces = int(line.split()[-1])
        if line.strip() == "end_header":
            break
        i += 1
    
    # parse vertices
    for v in lines[i+1 : i+1+num_vertices]:
        x, y, z, *rest = map(float, v.strip().split())
        vertices.append((x, y, z))
    
    # parse faces
    for f in lines[i+1+num_vertices : i+1+num_vertices+num_faces]:
        idx = list(map(int, f.strip().split()))
        n = idx[0]
        face_idx = idx[1:1+n]
        faces.append(face_idx)
    
    # create mesh in Blender
    mesh = bpy.data.meshes.new(obj_name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(obj_name, mesh)
    return obj


# === Load configuration ===
def load_camera_from_npz(npz_path, name="Camera_Main", image_size_x=None, image_size_y=None):
    data = np.load(npz_path)
    extrinsic = data["extrinsic"]  # 4x4 matrix
    angle_x = data["camera_angle_x"].item()

    # focal_px = float(data["focal"])            # original focal in pixels
    # render_w = bpy.context.scene.render.resolution_x
    # render_h = bpy.context.scene.render.resolution_y

    # focal_px_scaled = focal_px * (render_h / image_size_x)

    # compute vertical FoV (PyTorch3D expects vertical fov for FoV cameras)
    # fov = 2.0 * np.arctan(render_h / (2.0 * focal_px_scaled))

    cam_data = bpy.data.cameras.new(name)
    cam_data.lens_unit = 'FOV'
    cam_data.angle = angle_x

    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam_obj)

    # Set full 4x4 matrix
    mat = mathutils.Matrix(extrinsic)
    cam_obj.matrix_world = mat

    return cam_obj

# === Duplicate camera with rotation ===
def duplicate_camera_rotated(cam_obj, angle_deg=90, name="Camera_Rotated", offset=2.0):
    cam_copy = cam_obj.copy()
    cam_copy.data = cam_obj.data.copy()
    cam_copy.name = name
    bpy.context.collection.objects.link(cam_copy)

    # Rotate around origin in world space
    angle_rad = np.radians(angle_deg)
    rot = mathutils.Matrix.Rotation(angle_rad, 4, 'Z')
    cam_copy.matrix_world = rot @ cam_obj.matrix_world

    # Move backwards along camera's local Z axis (i.e., along its view direction)
    backward_vector = cam_copy.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, offset))
    cam_copy.location += backward_vector

    return cam_copy



# add white background to image with PIL 
def add_white_background(image_path):
    """Add white background to the rendered image using PIL."""

    # Open the image
    img = Image.open(image_path).convert("RGBA")

    # Create a new image with white background
    new_img = Image.new("RGBA", img.size, (255, 255, 255, 255))

    # Paste the original image onto the new image
    new_img.paste(img, (0, 0), img)

    # from segementation.py: 
    #img_rgb = cv2.convertScaleAbs(img_rgb, alpha=1.2, beta=0)
    # Save as PNG using Pillow for consistency
    # Image.fromarray(img_rgb).save(output_path, format="PNG")

    # add brightness like segmentation.py
    


    # Save the new image
    new_img.save(image_path.replace(".png", "_white_bg.png"), "PNG")


# === HDRI setup ===
def setup_hdri(hdri_path, rotation_deg=0, white_bg=False, as_hemisphere=False, strength=2.0):
    import numpy as np
    scene = bpy.context.scene

    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    world = scene.world
    world.use_nodes = True

    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    # if white background, skip HDRI entirely and just use solid color
    if white_bg:
        bg = nodes.new("ShaderNodeBackground")
        bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs["Strength"].default_value = 2.0
        output = nodes.new("ShaderNodeOutputWorld")
        links.new(bg.outputs["Background"], output.inputs["Surface"])
        print("Using solid white background (no HDRI)")
        return True

    # Check if HDRI file exists, if not use white background
    if not os.path.isfile(hdri_path):
        print(f"⚠️  HDRI file not found: {hdri_path}")
        print("   Using solid white background instead")
        bg = nodes.new("ShaderNodeBackground")
        bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs["Strength"].default_value = 2.0
        output = nodes.new("ShaderNodeOutputWorld")
        links.new(bg.outputs["Background"], output.inputs["Surface"])
        return True

    # === Nodes ===
    tex_coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    env_tex = nodes.new("ShaderNodeTexEnvironment")
    env_tex.image = bpy.data.images.load(hdri_path)

    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength
    output = nodes.new("ShaderNodeOutputWorld")

    # === Rotation setup ===
    mapping.inputs["Rotation"].default_value[2] = np.radians(rotation_deg)

    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], env_tex.inputs["Vector"])
    links.new(env_tex.outputs["Color"], bg.inputs["Color"])


    if as_hemisphere:
        # === Hemisphere lighting mask ===
        separate = nodes.new("ShaderNodeSeparateXYZ")
        math = nodes.new("ShaderNodeMath")
        math.operation = 'GREATER_THAN'
        math.inputs[1].default_value = 0.0  # Mask lower hemisphere

        black_bg = nodes.new("ShaderNodeBackground")
        black_bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        black_bg.inputs["Strength"].default_value = 0.0

        mix_shader = nodes.new("ShaderNodeMixShader")

        # View direction → Z axis
        links.new(tex_coord.outputs["Generated"], separate.inputs["Vector"])
        links.new(separate.outputs["Z"], math.inputs[0])

        links.new(math.outputs["Value"], mix_shader.inputs["Fac"])
        links.new(bg.outputs["Background"], mix_shader.inputs[2])      # top
        links.new(black_bg.outputs["Background"], mix_shader.inputs[1])  # bottom
        links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])
    else:
        links.new(bg.outputs["Background"], output.inputs["Surface"])
    return False


# === Render camera function ===
def render_camera(cam_obj, output_path, res_x, res_y, file_format) -> str:
    # set scene context
    scene = bpy.context.scene
    
    # set transparency
    scene.render.film_transparent = True
    
    scene.camera = cam_obj
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.image_settings.file_format = file_format
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

    return output_path + "." + file_format.lower()


# === Color management ===
def set_color_management(config: dict):
    scene = bpy.context.scene

    # Use standard view transform instead of Filmic
    scene.display_settings.display_device = 'sRGB'
    scene.view_settings.view_transform = config.get("view_transform", "Filmic")  # Default is 'Filmic'
    scene.view_settings.look = config.get("look", "Low Contrast")  # Default is 'High Contrast'
    scene.view_settings.exposure = config.get("exposure", 0.4)  # Default exposure
    scene.view_settings.gamma = config.get("gamma", 0.8)  # Default gamma

# === Add Plane/Grid on Ground for visualisation
def add_groundPlane():
    # set height to -1
    loc=(0,0,-0.75)   
    bpy.ops.mesh.primitive_plane_add(size=10, enter_editmode=False, align='WORLD', location=loc, rotation=(0, 0, 0))
    ground_plane = bpy.context.object
    ground_plane.name = "GroundPlane"
    bpy.ops.object.shade_smooth()
    return ground_plane


def create_scene_birdEye_cam(name="SceneCamera", radius=6.0, height=2.0,
                             angle_deg=0.0, fov=math.radians(45)):
    """
    Create a camera orbiting around the scene center at given radius, height and angle.
    
    - radius: distance from scene center in XY-plane
    - height: Z offset above scene center
    - angle_deg: rotation around Z axis in degrees (0 = +X, 90 = +Y)
    """
    # find scene center
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        center = mathutils.Vector((0,0,0))
    else:
        coords = [o.matrix_world @ mathutils.Vector(corner)
                  for o in meshes for corner in o.bound_box]
        center = sum(coords, mathutils.Vector()) / len(coords)
    
    # polar coordinates around Z
    angle = math.radians(angle_deg)
    eye = center + mathutils.Vector((radius*math.cos(angle),
                                     radius*math.sin(angle),
                                     height))
    
    # create camera
    cam_data = bpy.data.cameras.new(name + "_data")
    cam_data.lens_unit, cam_data.angle = 'FOV', fov
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam)
    
    # orient to look at center
    direction = (center - eye).normalized()
    cam.rotation_euler = direction.to_track_quat('-Z','Y').to_euler()
    cam.location = eye
    
    return cam

def setup_material(config: dict):
    """Creates the PBR material and assigns all four textures."""
    from PIL import Image
    
    mat = bpy.data.materials.new(name="PBR_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    
    base_mat_path = config.get("material_texture_path", "../output/findings/scene_marigold/")
    empty_room_path = os.path.join(config["output_inp_banana"], "empty_room.png")
    
    # Check if we should use simple baked image or full PBR
    use_baked_image = config.get("use_baked_image_only", True)

    # Get render resolution
    render_width = bpy.context.scene.render.resolution_x
    render_height = bpy.context.scene.render.resolution_y
    print(f" Rescaling textures to render resolution: {render_width}x{render_height}")
    
    # Get texture strength/scale factors from config
    roughness_strength = config.get("roughness_strength", 0.5)
    metallic_strength = config.get("metallic_strength", 0.1)
    normal_strength = config.get("normal_strength", 0.2)


    
    def rescale_texture(texture_path: str) -> str:
        """Helper to rescale texture to render resolution."""
        if not os.path.exists(texture_path):
            print(f"⚠️ Texture not found: {texture_path}")
            return None
            
        img = Image.open(texture_path)
        if img.size != (render_width, render_height):
            print(f" Rescaling {os.path.basename(texture_path)} from {img.size[0]}x{img.size[1]} to {render_width}x{render_height}")
            img = img.resize((render_width, render_height), Image.Resampling.LANCZOS)
            rescaled_path = texture_path.replace(".png", "_rescaled.png")
            img.save(rescaled_path)
            return rescaled_path
        return texture_path


    if use_baked_image:
        # Simple mode: just use the baked empty_room image as albedo
        print("🎨 Using baked image as albedo only (no PBR textures)")
        albedo_path = rescale_texture(empty_room_path)
        if albedo_path:
            albedo_tex = nodes.new(type="ShaderNodeTexImage")
            albedo_tex.image = bpy.data.images.load(albedo_path)
            mat.node_tree.links.new(albedo_tex.outputs["Color"], bsdf.inputs["Base Color"])
            # set to sRGB
            albedo_tex.image.colorspace_settings.name = 'sRGB'
        
        # Set default PBR values using strength sliders
        bsdf.inputs["Roughness"].default_value = roughness_strength
        bsdf.inputs["Metallic"].default_value = metallic_strength
        
    else:
        # Full PBR mode: use all texture maps
        print("🎨 Using full PBR textures (albedo, roughness, metallic, normal)")
        
        # --- Albedo (Base Color) ---
        albedo_path = rescale_texture(os.path.join(base_mat_path, "albedo_map.png"))
        if albedo_path:
            albedo_tex = nodes.new(type="ShaderNodeTexImage")
            albedo_tex.image = bpy.data.images.load(albedo_path)
            mat.node_tree.links.new(albedo_tex.outputs["Color"], bsdf.inputs["Base Color"])
            albedo_tex.image.colorspace_settings.name = 'sRGB'

        # --- Roughness ---
        roughness_path = rescale_texture(os.path.join(base_mat_path, "roughness_map.png"))
        if roughness_path:
            rough_tex = nodes.new(type="ShaderNodeTexImage")
            rough_tex.image = bpy.data.images.load(roughness_path)
            rough_tex.image.colorspace_settings.name = 'Non-Color'
            
            # Add ColorRamp/Math node to control strength
            if roughness_strength != 1.0:
                math_node = nodes.new(type="ShaderNodeMath")
                math_node.operation = 'MULTIPLY'
                math_node.inputs[1].default_value = roughness_strength
                mat.node_tree.links.new(rough_tex.outputs["Color"], math_node.inputs[0])
                mat.node_tree.links.new(math_node.outputs["Value"], bsdf.inputs["Roughness"])
            else:
                mat.node_tree.links.new(rough_tex.outputs["Color"], bsdf.inputs["Roughness"])

        # --- Metallic ---
        metallic_path = rescale_texture(os.path.join(base_mat_path, "metallic_map.png"))
        if metallic_path:
            metallic_tex = nodes.new(type="ShaderNodeTexImage")
            metallic_tex.image = bpy.data.images.load(metallic_path)
            metallic_tex.image.colorspace_settings.name = 'Non-Color'
            
            # Add Math node to control strength
            if metallic_strength != 1.0:
                math_node = nodes.new(type="ShaderNodeMath")
                math_node.operation = 'MULTIPLY'
                math_node.inputs[1].default_value = metallic_strength
                mat.node_tree.links.new(metallic_tex.outputs["Color"], math_node.inputs[0])
                mat.node_tree.links.new(math_node.outputs["Value"], bsdf.inputs["Metallic"])
            else:
                mat.node_tree.links.new(metallic_tex.outputs["Color"], bsdf.inputs["Metallic"])

        # --- Normal Map ---
        normal_path = rescale_texture(os.path.join(base_mat_path, "normal_map.png"))
        if normal_path:
            normal_tex = nodes.new(type="ShaderNodeTexImage")
            normal_tex.image = bpy.data.images.load(normal_path)
            normal_tex.image.colorspace_settings.name = 'Non-Color'
            
            normal_map_node = nodes.new(type="ShaderNodeNormalMap")
            normal_map_node.inputs["Strength"].default_value = normal_strength  # Control normal strength
            mat.node_tree.links.new(normal_tex.outputs["Color"], normal_map_node.inputs["Color"])
            mat.node_tree.links.new(normal_map_node.outputs["Normal"], bsdf.inputs["Normal"])
    
    return mat

def get_safe_3d_view_override():
    """
    Finds a safe and complete override context for running 3D View operators
    in headless (background) mode.
    
    Returns:
        A dictionary (override) for bpy.ops operators.
    Raises:
        RuntimeError: If a valid 3D View context cannot be found.
    """
    win, area, region, space = None, None, None, None
    
    # Iterate through all windows, screens, and areas
    for w in bpy.context.window_manager.windows:
        for a in w.screen.areas:
            if a.type == 'VIEW_3D':
                area = a
                # Find the main region for that area
                for r in a.regions:
                    if r.type == 'WINDOW':
                        region = r
                        break
                # Find the 3D space for that area
                for s in a.spaces:
                        if s.type == 'VIEW_3D':
                            space = s
                            break
            
            # If we found all parts, stop searching
            if area and region and space:
                win = w  # Store the window
                break
        if area and region and space:
            break  # Stop searching windows

    # Check if we successfully found everything
    if not (win and area and region and space):
        raise RuntimeError("Could not find a valid 3D Viewport context. Cannot project UVs.")

    # Build the complete override dictionary
    override = {
        'window': win,
        'screen': win.screen,
        'area': area,
        'region': region,
        'space': space,  # The crucial key
        'scene': bpy.context.scene,
    }
    
    return override


def main():
    # Load configuration
    # === Config defaults ===
    # Get the script directory first
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))  # Up two levels from src/blender_rendering/run.py
    
    # Try to load config from command-line argument or fallback to default
    config_path = None
    try:
        import sys
        # Check for --config argument
        if "--config" in sys.argv:
            config_idx = sys.argv.index("--config")
            if config_idx + 1 < len(sys.argv):
                config_path = sys.argv[config_idx + 1]
    except:
        pass
    
    if not config_path or not os.path.isfile(config_path):
        # Fallback: look for config.yaml in src folder
        default_config_path = os.path.join(repo_root, "src", "config.yaml")
        if os.path.exists(default_config_path):
            config_path = default_config_path
        else:
            # Try relative path as last resort
            default_config_path = os.path.abspath("../src/config.yaml")
            if os.path.exists(default_config_path):
                config_path = default_config_path
    
    if not config_path or not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    config = load_config(config_path)
    
    # Convert relative paths in config to absolute paths
    # This is important because Blender may change working directories
    paths_to_convert = [
        "output_render",
        "hdri_path",
        "glb_output_folder",
        "output_ply",
        "glb_scene_path",
        "ply_scene_bp_path",
        "input_images"
    ]
    
    for key in paths_to_convert:
        if key in config and config[key]:
            path_val = config[key]
            if not os.path.isabs(path_val):
                # Make it absolute relative to repo root
                # First, strip leading ../ from path_val to avoid going up from repo_root
                clean_path = path_val.lstrip("../")
                abs_path = os.path.join(repo_root, clean_path)
                config[key] = os.path.abspath(abs_path)

    output_dir = config.get("output_render", os.path.join(repo_root, "output/rendering/"))
    os.makedirs(output_dir, exist_ok=True)
    #tempfile.tempdir = output_dir
    render_pc = config.get("render_pointclouds", False)
    
    # addon_module_name = "io_mesh_ply"
    
    # # Path to the parent folder where addons are stored
    # addons_parent_dir = os.path.abspath("../blender/4.2/scripts/addons/")

    # # Check that the addon folder exists inside the addons directory
    # expected_addon_path = os.path.join(addons_parent_dir, addon_module_name)
    # if not os.path.exists(expected_addon_path):
    #     # NOTE: Your original path was ".../blender-addons/io_mesh_ply". 
    #     # If the folder is truly nested like that, Blender may not find it.
    #     # It's best for the `io_mesh_ply` folder to be directly inside `scripts/addons/`.
    #     raise FileNotFoundError(f"PLY addon module not found at: {expected_addon_path}")

    # # --- CORRECTED CODE ---
    # # The addon is already in the right place, so we just need to enable it.
    # # The 'install' step is not necessary and was causing the error.
    # print(f"Enabling addon: {addon_module_name}")
    # bpy.ops.preferences.addon_enable(module=addon_module_name)
    
    # # Optional but recommended: save preferences so it stays enabled
    # bpy.ops.wm.save_userpref()

    # print("Addon enabled successfully.")

    # clean scene
    # remove default cube from scene
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Cube'].select_set(True)
    bpy.ops.object.delete()

    # set rendering to cycles and set device to GPU
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'

    # Disable denoising to avoid the error
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.view_layers["ViewLayer"].cycles.use_denoising = True
    # set samples to 128
    bpy.context.scene.cycles.samples = config.get("blender_render_samples", 24)


    camera_npz = config.get("camera", "../output/pre_3D/camera.npz")
    # camera_emptyRoom = config.get("camera_emptyRoom", "../output/pre_3D/camera_emptyRoom.npz")

    # if os.path.exists(camera_emptyRoom):
    #     camera_npz = camera_emptyRoom
    
    if config["use_3d_front"] and not config.get("Use_VGGT", False):
        # append _dataset to camera_npz path name 
        camera_npz = camera_npz.replace(".npz", "_dataset.npz")
        print(f"Using 3D-FRONT camera data: {camera_npz}")

    #image_size_x = config.get("image_size", 768)
    # import original and get size
    image_path = config.get("input_image")
    image = Image.open(image_path)
    image_size_x, image_size_y = image.size

    # set render resolution to image size
    bpy.context.scene.render.resolution_x = image_size_x
    bpy.context.scene.render.resolution_y = image_size_y

    image.close()

    # === Cameras ===
    cam1 = load_camera_from_npz(camera_npz, "Camera_Main", image_size_x, image_size_y)
    if not config.get("Use_VGGT", False):
        cam2 = duplicate_camera_rotated(cam1, 90, "Camera_Rotated", offset=2.0)
    else:
        cam2 = create_scene_birdEye_cam(radius=2.0, height=1.0,
                             angle_deg=0.0)
    

    file_dir = config.get("out_pc_meshed", "../output/pointclouds/meshed/")
    file_path = os.path.join(file_dir, "ground_aligned.glb")
    
    if not os.path.exists(file_path):
        print(f"❌ Empty scene file not found: {file_path}")

    else: 
        bpy.ops.import_scene.gltf(filepath=file_path)
                
        # Get ALL selected objects from the import
        imported_objects = bpy.context.selected_objects

        mesh_obj = None # Initialize to be safe

        # Find the actual MESH object
        for obj in imported_objects:
            if obj.type == 'MESH':
                mesh_obj = obj
                break # Found it, stop looking

        # If no mesh was found, raise a clear error
        if mesh_obj is None:
            raise TypeError(f"No MESH object was found in the imported file: {file_path}")
            
        # Now, mesh_obj is guaranteed to be a mesh.
        bpy.context.view_layer.objects.active = mesh_obj
        
        # ⚠️ CRITICAL: Set the scene camera BEFORE UV projection
        # This ensures UVs are projected from the correct camera perspective
        bpy.context.scene.camera = cam1
        
        # setup material
        mat = setup_material(config)
        mesh_obj.data.materials.append(mat)
        # project uvs

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')

        try:
            print(f"Projecting UVs from camera: {cam1.name}")
            print(f"Render resolution: {image_size_x}x{image_size_y}")
            
            # Simple approach: just use the basic projection with all default settings
            # The key is that camera_bounds=True will use the render resolution
            override = get_safe_3d_view_override()
            
            with bpy.context.temp_override(**override):
                # Align viewport to camera
                bpy.ops.view3d.view_camera()
                
                # Use simplest projection settings that work
                # camera_bounds uses render resolution, scale_to_bounds fits to 0-1
                bpy.ops.uv.project_from_view(
                    camera_bounds=True,
                    correct_aspect=False,
                    scale_to_bounds=False,
                    clip_to_bounds=True
                )
            
            print("✅ UV projection completed")
            
        except RuntimeError as e:
            print(f"❌ Error during UV projection: {e}")
            # Handle the error as needed (e.g., exit or skip)
        
        bpy.ops.object.mode_set(mode='OBJECT')


    # === Load combined_scene.glb scene ===
    # create glb scene from config

    scene_path = config.get("glb_scene_path", 
                            os.path.join(config["glb_output_folder"], "combined_scene.glb")
                            )
    ply_file = config.get("ply_scene_bp_path", 
                            os.path.join(config["output_ply"], "combined_scene.ply")
                            )

    glb_file = scene_path

    # Check if glb_file exists, skip rendering if it doesn't
    if not os.path.exists(glb_file):
        print(f"⚠️  GLB scene file not found: {glb_file}")
        print("   Skipping blender rendering (required file not available)")
        print("   Note: Run previous pipeline steps (-p 1 through -p 7) to generate the scene")
        return

    if not os.path.exists(ply_file):
        print(f"⚠️  PLY scene file not found: {ply_file}")
        print("   Skipping point cloud rendering (PLY file not available)")


    # import glb scene
    bpy.ops.import_scene.gltf(filepath=glb_file)

    # === HDRI ===
    setup_hdri(
        config["hdri_path"],
        strength=config.get("hdri_strength", 2.0),
        rotation_deg=config.get("hdri_rotation", 0),
        white_bg=config.get("hdri_white_bg", True),
        as_hemisphere=False,
        )
    # === Color management ===
    set_color_management(config)

    # === Render each ===
    im1 = render_camera(
        cam1,
        os.path.join(output_dir, "render_cam1"),
        config.get("resolution_x", image_size_x),
        config.get("resolution_y", image_size_y),
        config.get("image_format", "PNG")
    )

    im2 = render_camera(
        cam2,
        os.path.join(output_dir, "render_cam2"),
        config.get("resolution_x", image_size_x),
        config.get("resolution_y", image_size_y),
        config.get("image_format", "PNG")
    )

    # === Add white background ===
    add_white_background(im1)
    add_white_background(im2)


    # === Render Point Clouds ===
    # load all glb files from config[output_ply] besides the one with scene
    #bpy.ops.wm.addon_install("~/.config/blender/4.0/scripts/addons/blender-addons/io_mesh_ply")
    #bpy.ops.wm.addon_install("../blender/4.2/scripts/addons/io_mesh_ply")

    # add method here
    if render_pc:    
        bpy.ops.preferences.addon_enable(module="io_mesh_ply")
        bpy.ops.import_mesh.ply(filepath=ply_file) # import point cloud
        # Get the last imported object (the point cloud)
        pc_obj = bpy.context.selected_objects[0]
        set_pc_for_render(pc_obj, config.get("pointcloud_scale", 0.002))


        im3 = render_camera(
            cam1,
            os.path.join(output_dir, "render_pointcloud_cam1"),
            config.get("resolution_x", image_size_x),
            config.get("resolution_y", image_size_y),
            config.get("image_format", "PNG")
        )
        im4 = render_camera(
            cam2,
            os.path.join(output_dir, "render_pointcloud_cam2"),
            config.get("resolution_x", image_size_x),
            config.get("resolution_y", image_size_y),
            config.get("image_format", "PNG")
        )

        add_white_background(im3)
        add_white_background(im4)


    # save blender scene to tmp/
    path = os.path.join("tmp/", "blender_scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)



    # Ground truth glb file
    GT_scene = config["GT_scene"]
    if GT_scene is not None and os.path.exists(GT_scene) and config.get("render_GT", False):
        scene_path = GT_scene

        if config["use_3d_front"]:
            input_image = config["input_image"]
            _3DFront_scene_dir = config.get("3d_front_scene", "../3D-Front/3D-FRONT-SCENE/")
            uuid = os.path.basename(os.path.dirname(os.path.dirname(input_image)))       
            scene_name = os.path.basename(os.path.dirname(input_image))                 

            # Compose full path to GLB
            scene_path = os.path.join(_3DFront_scene_dir, uuid, f"{scene_name}.glb")

        # clean scene completely and add correct glb and pointcloud
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type='MESH')
        bpy.ops.object.delete()

        # import ground truth glb scene
        bpy.ops.import_scene.gltf(filepath=scene_path)

        # === HDRI ===
        setup_hdri(
            config["hdri_path"], 
            rotation_deg=config.get("hdri_rotation", 0), 
            as_hemisphere=True
            )

        # === Color management ===
        set_color_management()

        # add method here
        if render_pc:
            # import predicted point cloud
            bpy.ops.import_mesh.ply(filepath=ply_file)

            # Get the last imported object (the point cloud)
            pc_obj = bpy.context.selected_objects[0]
            set_pc_for_render(pc_obj, config.get("pointcloud_scale", 0.002))

        # add groundPlane
        if config.get("add_groundPlane_BlenderVIS", False):
            add_groundPlane()

        im5 = render_camera(
            cam1,
            os.path.join(output_dir, "render_GT_PC_cam1"),
            config.get("resolution_x", image_size_x),
            config.get("resolution_y", image_size_y),
            config.get("image_format", "PNG")
        )
        im6 = render_camera(
            cam2,
            os.path.join(output_dir, "render_GT_PC_cam2"),
            config.get("resolution_x", image_size_x),
            config.get("resolution_y", image_size_y),
            config.get("image_format", "PNG")
        )

        add_white_background(im5)
        add_white_background(im6)

    if not config.get("silent", False):
        print("Rendering done.")


if __name__ == "__main__":
    main()
