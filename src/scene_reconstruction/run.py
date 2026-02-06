import argparse
import yaml
from source.pose_matching_planar import pose_matching
import os

from src.utils.global_utils import load_config
import torch
import multiprocessing as mp
import logging

def worker(config, model_name, iteration, device_id):
    """A wrapper function to run pose_matching on a specific GPU."""
    try:
        # Set the CUDA_VISIBLE_DEVICES environment variable for this specific process
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        logging.info(
            f"Starting process for '{model_name}' on GPU {device_id} (PID: {os.getpid()})"
        )

        # The pose_matching function will now see only the assigned GPU
        pose_matching(config, model_name=model_name, iteration=iteration, device_id=device_id)
        logging.info(f"Finished process for '{model_name}' on GPU {device_id}.")
    except Exception as e:
        logging.info(f"ERROR in worker for '{model_name}' on GPU {device_id}: {e}")


if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn' for CUDA compatibility
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Differential Rendering Runner")

    parser.add_argument(
        "--config",
        type=str,
        default="../src/config.yaml",
        help="Path to config file in YAML format",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Dynamically set logging level based on config['logging']
    level_str = config.get("logging", "INFO").upper()  # Default to 'INFO' if not set
    level = getattr(
        logging, level_str, logging.INFO
    )  # Map string to logging level (e.g., 'DEBUG' -> logging.DEBUG)
    logging.getLogger().setLevel(level)
    logging.info(f"Logging level set to: {level_str}")

    # get "cropped" folder path from config
    cropped_folder = config.get("full_size", None)
    if cropped_folder is None:
        raise ValueError("Cropped folder path not found in config file.")

    # set ignore classes for pose matching
    ignore_classes = config.get(
        "ignore_classes", ["wall", "floor", "ceiling", "door", "window"]
    )

    # --- Prepare list of tasks ---
    glb_root = config.get("output_folder_hy", None)
    if glb_root is None:
        raise ValueError("GLB path not provided in config.")
    tasks = []
    for i, image_name_ext in enumerate(os.listdir(cropped_folder)):
        if any(ignore_class in image_name_ext for ignore_class in ignore_classes):
            continue

        image_name = os.path.splitext(image_name_ext)[0]
        glb_path = os.path.join(glb_root, image_name, f"{image_name}.glb")
        if not os.path.exists(glb_path):
            logging.warning(f"Skipping {image_name}: missing GLB at {glb_path}")
            continue
        logging.info(f"Prepared {image_name} tasks for processing.")
        tasks.append((config, image_name, i))
    
    

    # --- Run sequentially or in parallel based on config ---
    if config.get("use_all_available_cuda", True):
        num_devices = torch.cuda.device_count()
        max_jobs_per_gpu = int(config.get("max_jobs_per_gpu", 3))  # Default to 4 if not set
        total_jobs = num_devices * max_jobs_per_gpu if num_devices > 0 else 1
        if num_devices > 0:
            logging.info(f"Found {num_devices} GPUs. Running up to {max_jobs_per_gpu} jobs per GPU, total pool size: {total_jobs}.")
            # Assign a device_id to each task, cycling through available GPUs
            tasks_with_devices = [
                (cfg, name, it, idx % num_devices)
                for idx, (cfg, name, it) in enumerate(tasks)
            ]

            # Use a process pool to execute tasks (multiple jobs per GPU)
            with mp.Pool(processes=total_jobs) as pool:
                pool.starmap(worker, tasks_with_devices)
            logging.info("All parallel tasks completed.")

        else:
            logging.info(
                "Parallel execution requested, but no GPUs found. Running sequentially."
            )
            for cfg, name, it in tasks:
                pose_matching(cfg, model_name=name, iteration=it, device_id=0)
    else:
        logging.info("Running sequentially on default device.")
        for cfg, name, it in tasks:
            pose_matching(cfg, model_name=name, iteration=it, device_id=0)
