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
    """Processes a single image to generate and save a 3D model."""
    image = Image.open(image_path).convert("RGBA")
    if image.mode == "RGB":
        image = rembg(image)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"Processing {base_name}...")
    start_time = time.time()

    mesh = pipeline_shapegen(
        image=image,
        num_inference_steps=config.get("num_inf_steps_hy", 100),
        octree_resolution=config.get("octree_resolution_hy", 380),
        num_chunks=config.get("num_chunks_hy", 20000),
        generator=torch.manual_seed(config.get("seed", 12345)),
        output_type="trimesh"
    )[0]

    if config.get("remesh", False):
        print("Remeshing enabled. Cleaning and simplifying the mesh...")
        target_faces = config.get("remesh_target_num_faces", 30000)
        mesh = clean_and_validate_trimesh(mesh, target_face_count=target_faces)
        
    print(f"Initial mesh has {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")

    for cleaner in [FloaterRemover(), DegenerateFaceRemover(), FaceReducer()]:
        mesh = cleaner(mesh)
    print(f"Cleaned mesh has {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")

    mesh = pipeline_texgen(mesh, image=image)

    out_dir = os.path.join(output_dir, base_name)
    os.makedirs(out_dir, exist_ok=True)
    out_mesh_path = os.path.join(out_dir, f"{base_name}.glb")
    mesh.export(out_mesh_path)

    duration = time.time() - start_time
    print(f"Saved {base_name} to {out_mesh_path} in {duration:.2f} seconds.")


def worker(image_path, output_folder, config, device_id, shapegen_config, texgen_config):
    """
    Worker process: isolates a GPU, initializes models, and processes one image.
    """
    try:
        # Isolate the GPU for this worker process
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        device = 'cuda:0' # The worker will only see this one GPU
        print(f"Worker (PID: {os.getpid()}) processing '{os.path.basename(image_path)}' on GPU {device_id}")

        # Initialize models within the worker. This is necessary for 'spawn' multiprocessing.
        shapegen_path = snapshot_download(repo_id=shapegen_config["id"])
        texgen_path = snapshot_download(repo_id=texgen_config["id"])
        
        pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            shapegen_path, **shapegen_config["args"]
        )
        
        pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(
            texgen_path, **texgen_config["args"]
        )

        rembg = BackgroundRemover()

        process_image(image_path=image_path, pipeline_shapegen=pipeline_shapegen, pipeline_texgen=pipeline_texgen, output_dir=output_folder, rembg=rembg, config=config)
        print(f"Worker (PID: {os.getpid()}) finished '{os.path.basename(image_path)}'.")

    except Exception as e:
        print(f"ERROR in worker for '{os.path.basename(image_path)}' on GPU {device_id}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run 2D to 3D model generation.")
    parser.add_argument("--config", default="../src/config.yaml", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    config = load_config(args.config)

    # --- Define Model Configurations ---
    shapegen_models = {
        "full": {"id": "tencent/Hunyuan3D-2", "args": {}},
        "mini": {"id": "tencent/Hunyuan3D-2mini", "args": {"subfolder": "hunyuan3d-dit-v2-mini", "variant": "fp16"}}
    }
    # Texture generator is always the same, as per your example
    texgen_model = {"id": "tencent/Hunyuan3D-2", "args": {}}
    
    # Select which shape generation model to use
    use_mini = config.get("mini", True)
    shapegen_key = "mini" if use_mini else "full"
    selected_shapegen_config = shapegen_models[shapegen_key]
    print(f"Using '{shapegen_key}' shape generator: {selected_shapegen_config['id']}")

    # --- File and Folder Setup ---
    input_folder = config["input_folder_hy"]
    if config["use_banana"]:
        input_folder = config["prepped_for_hunyuan"]

    output_folder = config["output_folder_hy"]
    os.makedirs(output_folder, exist_ok=True)
    clear_output_directory(output_folder)

    # no images with names containing : walls, room, ceiling, floor
    image_extensions = (".png", ".jpg", ".jpeg")
    dont_mesh = ["wall", "walls", "room", "ceiling", "floor"]
    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(image_extensions) and not any(x in f.lower() for x in dont_mesh)]
    if not image_paths:
        raise FileNotFoundError(f"No images found in the input folder '{input_folder}'.")

    # --- Execution Logic ---
    num_devices = torch.cuda.device_count()
    jobs_per_gpu = int(config.get("jobs_per_gpu", 1))
    if jobs_per_gpu < 1:
        jobs_per_gpu = 1

    total_slots = num_devices * jobs_per_gpu

    if num_devices >= 1 and len(image_paths) > 1 and total_slots > 1:
        processes = min(total_slots, len(image_paths))
        print(
            f"Found {num_devices} GPU(s). Using up to {jobs_per_gpu} job(s) per GPU -> {processes} parallel worker(s)."
        )
        tasks = [
            (path, output_folder, config, i % num_devices, selected_shapegen_config, texgen_model)
            for i, path in enumerate(image_paths)
        ]
        with mp.Pool(processes=processes) as pool:
            pool.starmap(worker, tasks)
        print("All parallel tasks completed.")

    else:
        device_msg = (
            "no GPU found" if num_devices == 0 else "only 1 slot or only 1 image"
        )
        print(f"Running sequentially ({device_msg}).")
        shapegen_path = snapshot_download(repo_id=selected_shapegen_config["id"])
        texgen_path = snapshot_download(repo_id=texgen_model["id"])
        
        pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            shapegen_path, **selected_shapegen_config["args"]
        )
        pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(
            texgen_path, **texgen_model["args"]
        )
        rembg = BackgroundRemover()

        for image_path in image_paths:
            process_image(image_path, pipeline_shapegen, pipeline_texgen, output_folder, rembg, config)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

