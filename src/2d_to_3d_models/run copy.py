import os
import time
import argparse
import torch
from PIL import Image
import yaml
import multiprocessing as mp

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import (
    Hunyuan3DDiTFlowMatchingPipeline,
    FaceReducer,
    FloaterRemover,
    DegenerateFaceRemover,
)
from hy3dgen.texgen import Hunyuan3DPaintPipeline

from utils.global_utils import clear_output_directory, load_config
import trimesh
import numpy as np
from huggingface_hub import snapshot_download

def clean_and_validate_trimesh(mesh, min_faces=10, target_face_count=None):
    """
    Cleans, validates, and optionally remeshes a trimesh object.

    Args:
        mesh (trimesh.Trimesh): The input mesh.
        min_faces (int): The minimum number of faces the mesh must have to be considered valid.
        target_face_count (int, optional): If provided, simplifies the mesh to this number of faces.

    Returns:
        trimesh.Trimesh or None: The cleaned and simplified mesh.
    """
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError("Input is not a valid or is an empty trimesh object.")

    # 1. Remove NaN/Inf vertices
    valid_verts_mask = np.all(np.isfinite(mesh.vertices), axis=1)
    if not np.all(valid_verts_mask):
        num_invalid = (~valid_verts_mask).sum()
        print(f"[WARN] Found {num_invalid} invalid (NaN/Inf) vertices. Cleaning...")
        mesh.update_vertices(valid_verts_mask)

    # 2. **NEW**: Simplify the mesh to the target face count if specified
    if target_face_count is not None and len(mesh.faces) > target_face_count:
        print(f"Simplifying mesh from {len(mesh.faces)} to {target_face_count} faces...")
        mesh = mesh.simplify_quadric_decimation(face_count=target_face_count)
        print(f"Simplified mesh has {len(mesh.faces)} faces.")

    # 3. Check if the mesh is now empty or below the minimum threshold
    if mesh.is_empty or len(mesh.faces) < min_faces:
        raise ValueError(f"Mesh is empty or has fewer than {min_faces} faces after cleaning/simplification.")

    # 4. Use trimesh's built-in repair functions
    mesh.process(validate=True)
    mesh.remove_unreferenced_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())

    if mesh.is_empty or len(mesh.faces) < min_faces:
        raise ValueError("Mesh became empty after final processing.")

    return mesh


def process_image(image_path, pipeline_shapegen, pipeline_texgen, output_dir, rembg, config):
    # Open image and convert to RGBA
    image = Image.open(image_path).convert("RGBA")
    # Optionally remove the background if image is RGB (or if desired)
    if image.mode == "RGB":
        image = rembg(image)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"Processing {base_name}...")

    start_time = time.time()

    # Create mesh using the shapegen pipeline
    mesh = pipeline_shapegen(
        image=image,
        num_inference_steps=config.get("num_inf_steps_hy", 100),
        octree_resolution=config.get("octree_resolution_hy", 380),
        num_chunks= config.get("num_chunks_hy", 20000),
        generator=torch.manual_seed(config.get("seed", 12345)),
        output_type="trimesh"
    )[0]
    # Clean and validate the mesh
    mesh = clean_and_validate_trimesh(mesh, target_face_count=30000)
    print(f"Initial mesh has {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")
    # Clean mesh with a series of removers
    for cleaner in [FloaterRemover(), DegenerateFaceRemover(), FaceReducer()]:
        mesh = cleaner(mesh)

    print(f"Cleaned mesh has {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")
    # === ADDED: Final Validation and Cleaning Step ===
    

    # Generate texture using the texgen pipeline
    mesh = pipeline_texgen(mesh, image=image)

    # Create output subfolder (e.g. output_dir/model123)
    out_dir = os.path.join(output_dir, base_name)
    os.makedirs(out_dir, exist_ok=True)
    out_mesh_path = os.path.join(out_dir, f"{base_name}.glb")
    mesh.export(out_mesh_path)

    duration = time.time() - start_time
    print(f"Saved {base_name} to {out_mesh_path} in {duration:.2f} seconds.")



def worker(image_path, output_folder, config, device_id):
    """
    Worker process to handle a single image on a specific GPU.
    Initializes models and processes the image.
    """
    try:
        # 1. Set the visible device for this process
        device = f"cuda:{device_id}"
        torch.cuda.set_device(device)
        print(f"Worker (PID: {os.getpid()}) processing '{os.path.basename(image_path)}' on GPU {device_id}")

        # 2. Initialize models within the worker process
        local_model_path = snapshot_download(repo_id='tencent/Hunyuan3D-2')
        
        pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            local_model_path
        ).to(device)
        
        pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(
            local_model_path
        ).to(device)

        rembg = BackgroundRemover(device=device)

        # 3. Call the original processing function
        process_image(
            image_path,
            pipeline_shapegen,
            pipeline_texgen,
            output_folder,
            rembg,
            config=config
        )
        print(f"Worker (PID: {os.getpid()}) finished '{os.path.basename(image_path)}'.")

    except Exception as e:
        print(f"ERROR in worker for '{os.path.basename(image_path)}' on GPU {device_id}: {e}")





def main():
    # Load configuration
    parser = argparse.ArgumentParser(description="Run segmentation script with config file.")
    parser.add_argument("--config", default="../src/config.yaml", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)

    input_folder = config["input_folder_hy"]
    output_folder = config["output_folder_hy"]

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    # Clear output directory if it exists
    clear_output_directory(output_folder)

    mini = config.get("mini", True)

    # Get list of images (png, jpg, jpeg) from input folder
    image_extensions = (".png", ".jpg", ".jpeg")
    image_paths = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(image_extensions)
    ]
    if not image_paths:
        raise FileNotFoundError(f"No images found in the input folder to 3D-ize'{input_folder}'.")

    # # Initialize the background remover
    rembg = BackgroundRemover()

    # # # Load shape-generation and texture-generation pipelines
    # # model_path = "tencent/Hunyuan3D-2"
    # # if mini:
    # #     pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    # #         model_path="tencent/Hunyuan3D-2mini",
    # #         subfolder="hunyuan3d-dit-v2-mini",
    # #         variant="fp16"
    # #     )
    # # else:
    # #     pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    # #         model_path=model_path,
    # #     )
    
    # model_path = 'tencent/Hunyuan3D-2.1'
    # pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    #     model_path,
    #     subfolder='hunyuan3d-dit-v2-1',
    #     ckpt_name='model.fp16.ckpt'
    # )
    
    
    
    # --- Parallel Execution Setup ---
    num_devices = torch.cuda.device_count()
    if num_devices > 1:
        print(f"Found {num_devices} GPUs. Distributing tasks...")
        
        # Prepare tasks for the workers
        tasks = []
        for i, image_path in enumerate(image_paths):
            device_id = i % num_devices
            tasks.append((image_path, output_folder, config, device_id))
        
        # Use a process pool to execute tasks in parallel
        with mp.Pool(processes=num_devices) as pool:
            pool.starmap(worker, tasks)
        
        print("All parallel tasks completed.")

    else:
        # --- Fallback to Sequential Execution ---
        print("Found 1 or 0 GPUs. Running sequentially.")
        # Initialize models once for sequential run
        local_model_path = snapshot_download(repo_id='tencent/Hunyuan3D-2')
        print("Loading shape generation pipeline...")
        pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(local_model_path)
        print("Loading texture generation pipeline...")
        pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(local_model_path)
        rembg = BackgroundRemover()

        for image_path in image_paths:
            process_image(image_path,
                          pipeline_shapegen,
                          pipeline_texgen,
                          output_folder, 
                          rembg,
                          config=config)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
