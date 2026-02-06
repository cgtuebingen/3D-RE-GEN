import os
import sys
import time
import torch
from PIL import Image
import trimesh

# Add Hunyuan3D-2.1 modules to path FIRST (since cwd is set by run.py) 
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

# Add the main project utils to path AFTER hy3dpaint to avoid namespace conflict
sys.path.insert(0, '../src')
# Import global utils with explicit module path to avoid conflict
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("global_utils", "../src/utils/global_utils.py")
global_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(global_utils)
clear_output_directory = global_utils.clear_output_directory
load_config = global_utils.load_config

from hy3dshape.rembg import BackgroundRemover  
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.pipelines import export_to_trimesh
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

TEXTURE_AVAILABLE = True

# Apply torchvision fix
try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")                                      
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")


def export_mesh(mesh, save_folder, name, textured=False):
    """Export mesh as GLB only"""
    if textured:
        path = os.path.join(save_folder, f'{name}.glb')
    else:
        path = os.path.join(save_folder, f'{name}_shape.glb')
    mesh.export(path)
    return path

def process_image(image_path, output_dir, config):
    """Process a single image using Hunyuan3D-2.1 pipeline (following gradio_app.py pattern)."""
    
    # Load image
    image = Image.open(image_path).convert("RGBA")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    print(f"Processing {base_name}...")
    
    # Remove background if needed
    if config.get("check_box_rembg", True) or image.mode == 'RGB':
        rembg = BackgroundRemover()
        image = rembg(image.convert('RGB'))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize shape generation pipeline
    model_path = 'tencent/Hunyuan3D-2.1'
    pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
    
    # Get generation parameters from config
    steps = config.get("steps_hy21", 5)
    guidance_scale = config.get("guidance_scale_hy21", 7.5)
    seed = config.get("seed", 1234)
    octree_resolution = config.get("octree_resolution_hy21", 256)
    num_chunks = config.get("num_chunks_hy21", 200000)
    
    # Generate shape
    print(f"Generating shape for {base_name} (steps={steps}, guidance={guidance_scale}, seed={seed})...")
    shape_start_time = time.time()
    
    generator = torch.Generator()
    generator = generator.manual_seed(int(seed))
    
    outputs = pipeline_shapegen(
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        octree_resolution=octree_resolution,
        num_chunks=num_chunks,
        output_type='mesh'
    )
    
    # Export to trimesh
    mesh = export_to_trimesh(outputs)[0]
    shape_path = export_mesh(mesh, output_dir, base_name, textured=False)
    
    shape_time = time.time() - shape_start_time
    print(f"Shape generation completed in {shape_time:.2f} seconds")
    
    # Generate texture if enabled
    if config.get("enable_texture_hy21", True):
        print(f"Generating texture for {base_name}...")
        texture_start_time = time.time()
        
        # Initialize paint pipeline with config parameters
        max_num_view = config.get("max_num_view_hy21", 8)  # can be 6 to 9
        resolution = config.get("resolution_hy21", 512)  # can be 768 or 512
        conf = Hunyuan3DPaintConfig(max_num_view, resolution)
        conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
        conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
        paint_pipeline = Hunyuan3DPaintPipeline(conf)
        
        # Apply texture directly to GLB
        textured_glb_path = os.path.join(output_dir, f"{base_name}.glb")
        output_mesh_path = paint_pipeline(
            mesh_path=shape_path,
            image_path=image_path,
            output_mesh_path=textured_glb_path,
            save_glb=True  # Force GLB output
        )
        
        texture_time = time.time() - texture_start_time
        print(f"Texture generation completed in {texture_time:.2f} seconds")
        print(f"Textured GLB saved: {textured_glb_path}")
    
    total_time = time.time() - shape_start_time
    print(f"Total processing time for {base_name}: {total_time:.2f} seconds")


def main():
    """Main function following the pattern of the original run.py"""
    
    # Load configuration (everything controlled by config)
    config = load_config("../src/config.yaml")
    
    # --- File and Folder Setup (same as original) ---
    input_folder = config["input_folder_hy"]
    if config.get("use_banana"):
        input_folder = config["prepped_for_hunyuan"]

    output_folder = config["output_folder_hy"]
    os.makedirs(output_folder, exist_ok=True)
    clear_output_directory(output_folder)

    image_extensions = (".png", ".jpg", ".jpeg")
    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]
    if not image_paths:
        raise FileNotFoundError(f"No images found in the input folder '{input_folder}'.")
    
    print(f"Found {len(image_paths)} images to process")

    # Process each image sequentially
    print("Processing images sequentially...")

    for image_path in image_paths:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join(output_folder, base_name)
        process_image(image_path, output_dir, config)
        # clean cuda cache after each image
        torch.cuda.empty_cache()

    print(f"Processing complete. Results saved to {output_folder}")


if __name__ == "__main__":
    main()