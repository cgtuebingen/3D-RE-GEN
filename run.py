import logging
import subprocess
import os
import sys
from tqdm import tqdm
import argparse

# import time
import time

from src.utils.global_utils import load_config


def find_python_executable(venv_path, use_conda_env=None):
    """
    Find Python executable. If use_conda_env is specified, use that conda environment.
    Otherwise, look for a venv at venv_path.
    
    Args:
        venv_path: Path to virtual environment (ignored if use_conda_env is set)
        use_conda_env: Name or path to conda environment (e.g., "py-310-fresh" or full path)
    """
    # If conda environment is specified, use it
    if use_conda_env:
        # Check if it's a full path or just an environment name
        if os.path.isabs(use_conda_env):
            conda_bin = os.path.join(use_conda_env, "bin", "python")
        else:
            # Try to find conda and locate the environment
            conda_prefix = os.environ.get("CONDA_PREFIX_1") or os.environ.get("CONDA_EXE")
            if conda_prefix:
                if conda_prefix.endswith("bin/conda"):
                    conda_base = os.path.dirname(os.path.dirname(conda_prefix))
                else:
                    conda_base = conda_prefix
                conda_bin = os.path.join(conda_base, "envs", use_conda_env, "bin", "python")
            else:
                # Fallback: use the provided path directly
                conda_bin = os.path.join(use_conda_env, "bin", "python")
        
        if os.path.isfile(conda_bin):
            return conda_bin
        else:
            raise FileNotFoundError(f"Conda Python executable not found at {conda_bin}")
    
    # Original venv logic
    bin_dir = os.path.join(venv_path, "Scripts" if sys.platform == "win32" else "bin")
    candidates = (
        ["python.exe"]
        if sys.platform == "win32"
        else ["python", "python3", "python3.12"]
    )

    for candidate in candidates:
        exe_path = os.path.join(bin_dir, candidate)
        if os.path.isfile(exe_path):
            return exe_path
    raise FileNotFoundError(f"No Python executable found in {bin_dir}")


def run_script(
    venv_path,
    script_path,
    cwd,
    use_blender=False,
    device="cpu",
    allow_all_gpus=False,
    use_conda_env=None,
    args=None,
):
    python_executable = find_python_executable(venv_path, use_conda_env=use_conda_env)
    env = os.environ.copy()
    # Set PYTHONPATH to the current working directory so that the inner dust3r folder is on sys.path
    env["PYTHONPATH"] = cwd
    # Add src paths for both repo-root and submodule execution contexts
    candidate_src_paths = [
        os.path.join(cwd, "src"),
        os.path.join(cwd, "..", "src"),
        os.path.join(cwd, "..", "src", "utils"),
        os.path.join(cwd, "src", "utils"),
    ]
    for src_path in candidate_src_paths:
        if os.path.isdir(src_path):
            env["PYTHONPATH"] += os.pathsep + src_path

    # Selectively set CUDA_VISIBLE_DEVICES for the subprocess
    if not allow_all_gpus and device.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = device.split(":")[1]

    if use_blender:
        try:
            # Get Python version from the venv
            version_result = subprocess.run(
                [python_executable, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                capture_output=True,
                text=True,
                check=True
            )
            venv_python_version = version_result.stdout.strip()
            
            if venv_python_version == "3.12":
                blender_executable = "blender"
            else:
                # Use absolute path to blender executable
                blender_executable = os.path.join(os.path.dirname(script_dir), "blender", "blender")
                
        except subprocess.CalledProcessError:
            # Fallback to system blender if we can't determine version
            print("Warning: Could not determine venv Python version, using system blender")
            blender_executable = "blender"
        
        command = [blender_executable, "-b", "-P", script_path]

    else:
        python_executable = find_python_executable(venv_path)
        command = [python_executable, script_path]
        if args:
            command.extend(args)

    subprocess.run(command, cwd=cwd, check=True, env=env)


def run_all(
    scripts_to_run,
    parts_to_run=None,
    exclude_parts=None,
    start_time=None,
    device="cpu",
    config=None,
    use_conda_env=None,
):
    if parts_to_run is None:
        if exclude_parts is None:
            # Default: run all scripts
            parts_to_run = list(range(len(scripts_to_run)))
        else:
            # Convert exclude_parts to zero-based indices
            exclude_parts_zero = [ex - 1 for ex in exclude_parts]

            # Validate excluded parts
            for ex in exclude_parts_zero:
                if ex < 0 or ex >= len(scripts_to_run):
                    raise ValueError(f"Invalid exclude part number: {ex + 1}")

            # Generate parts_to_run as all indices not in exclude_parts_zero
            parts_to_run = [
                i for i in range(len(scripts_to_run)) if i not in exclude_parts_zero
            ]
    else:
        # Convert to zero-based indices
        parts_to_run = [part - 1 for part in parts_to_run]
        # Validate the parts
        for part in parts_to_run:
            if part < 0 or part >= len(scripts_to_run):
                raise ValueError(f"Invalid part number: {part + 1}")

    print("Running scripts with progress bar...\n")

    for idx in parts_to_run:
        task = scripts_to_run[idx]
        print(f"\n▶ Running {task['name']} ...")
        temp_time = time.time()

        # --- MODIFIED: Check if this task should see all GPUs ---
        allow_all_gpus = False
        # Add the 2d_to_3d_models script to the list of scripts that can use multiple GPUs
        parallel_scripts = ["scene_reconstruction", "2d_to_3d_models", "run_hunyuan21", "segmentation"]
        if any(name in task["script"] for name in parallel_scripts) and config.get(
            "use_all_available_cuda", False
        ):
            allow_all_gpus = True
            print(
                "INFO: Allowing subprocess to access all available GPUs for parallel processing."
            )
        # --- END MODIFICATION ---

        try:
            # Check if this specific task should use conda env
            task_conda_env = None
            if task.get("use_conda", False) and use_conda_env:
                task_conda_env = use_conda_env
                print(f"   Using conda environment: {task_conda_env}")
            
            run_script(
                venv_path=task["venv"],
                script_path=task["script"],
                cwd=task["cwd"],
                use_blender=task.get("use_blender", False),
                device=device,
                allow_all_gpus=allow_all_gpus,
                use_conda_env=task_conda_env,
                args=task.get("args"),
            )

            print(f"✅ Finished {task['name']}")
            (
                print(f"Time taken: {(time.time() - temp_time)/60} minutes\n")
                if start_time
                else None
            )

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to run {task['name']}: {e}")
            # Stop
            break

        except FileNotFoundError as e:
            print(f"⚠️  Missing Python executable for {task['name']}: {e}")


def get_scripts(*pipeline_types: list) -> list:
    """
    Get scripts for one or more pipeline types.

    Args:
        pipeline_types (str): A single pipeline type (e.g., "segmentation") or
                              a comma-separated list (e.g., "segmentation,camera_dust3r").

    Returns:
        list: Concatenated list of script dictionaries for the specified types.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    main_venv = os.path.join(script_dir, "venv_py310")
    pipelines = {
        "segmentation": [
            {
                "name": "Grounded SAM and upscaling : (segmentor folder)",
                "venv": main_venv,
                "script": os.path.join(
                    script_dir, "src", "segmentation", "segmentation.py"
                ),
                "cwd": os.path.join(script_dir, "segmentor"),
            },
        ],
        "inpainting": [
            {
                "name": "Inpainting with nano banana : (segmentor folder)",
                "venv": main_venv,
                "script": os.path.join(
                    script_dir, "src", "segmentation", "inpaint_nanoBanana.py"
                ),
                "cwd": os.path.join(script_dir, "segmentor"),
            },
        ],
        "camera_dust3r": [
            {
                "name": "Extracting Camera and point cloud : (dust3r folder)",
                "venv": main_venv,
                "script": os.path.join(
                    script_dir, "src", "camera_and_pointcloud", "minimal_demo_dust3r.py"
                ),
                "cwd": os.path.join(script_dir, "dust3r"),
            },
        ],
        "Hunyuan_2d_to_3d": [
            {
                "name": "2D cropped images to 3D models : (Hunyuan3D-2.0 folder)",
                "venv": main_venv,
                "script": os.path.join(script_dir, "src", "2d_to_3d_models", "run.py"),
                "cwd": os.path.join(script_dir, "src"),
                "args": ["--config", os.path.join(script_dir, "src", "config.yaml")],
            },
        ],
        "Hunyuan_2d_to_3d_v21": [
            {
                "name": "2D cropped images to 3D models : (Hunyuan3D-2.1 folder)",
                "venv": main_venv,
                "script": os.path.join(script_dir, "src", "2d_to_3d_models", "run_hunyuan21.py"),
                "cwd": os.path.join(script_dir, "src"),
                "args": ["--config", os.path.join(script_dir, "src", "config.yaml")],
            },
        ],
        "point_cloud_extraction": [
            {
                "name": "Point Cloud Extraction : (Pytorch3D folder)",
                "venv": main_venv,
                "script": os.path.join(
                    script_dir, "src", "scene_reconstruction", "source", "extract_pc_object.py"
                ),
                "cwd": os.path.join(script_dir, "src"),
                "args": ["--config", os.path.join(script_dir, "src", "config.yaml")],
                #"use_conda": True,  # Use conda env for this task
            },
        ],
        "scene_recon": [
            {
                "name": "3D Scene reconstruction : (Pytorch3D folder)",
                "venv": main_venv,
                "script": os.path.join(
                    script_dir, "src", "scene_reconstruction", "run.py"
                ),
                "cwd": os.path.join(script_dir, "src"),
                "args": ["--config", os.path.join(script_dir, "src", "config.yaml")],
            },
        ],
        "scene_optim": [
            {
                "name": " 3D scene optimization : (src/scene_optimization folder)",
                "venv": main_venv,
                "script": os.path.join(
                    script_dir, "src", "scene_optimization", "scene_optim.py"
                ),
                "cwd": os.path.join(script_dir, "src"),
                "args": ["--config", os.path.join(script_dir, "src", "config.yaml")],
            },
        ],
        "render": [
            {
                "name": "Render scene in blender : (blender folder)",
                # Not used for Blender
                "venv": main_venv,
                "script": os.path.join(
                    script_dir, "src", "blender_rendering", "run.py"
                ),
                "cwd": os.path.join(script_dir, "src"),
                "args": ["--config", os.path.join(script_dir, "src", "config.yaml")],
                "use_blender": True,
            },
        ],
        "evaluation": [
            {
                "name": "Evaluate 3D models with MIDI metrics : (evaluation folder)",
                "venv": main_venv,
                "script": os.path.join(script_dir, "src", "evaluation", "run_eval.py"),
                "cwd": os.path.join(script_dir, "src"),
                "args": ["--config", os.path.join(script_dir, "src", "config.yaml")],
            },
        ],
        "camera_vggt": [
            {
                "name": "Extracting Camera and point cloud with vggt",
                "venv": os.path.join(script_dir, "vggt", ".venv"),
                "script": os.path.join(
                    script_dir, "src", "camera_and_pointcloud", "minimal_demo_vggt.py"
                ), # "minimal_demo_vggt_unproject.py"
                "cwd": os.path.join(script_dir, "vggt"),
            },
        ],
        "MIDI_2d_to_3d": [
            {
                "name": "2D to 3D with MIDI",
                "venv": main_venv,
                "script": os.path.join(script_dir, "src", "evaluation", "run_midi.py"),
                "cwd": os.path.join(script_dir, "src"),
                "args": ["--config", os.path.join(script_dir, "src", "config.yaml")],
            },
        ],
        "DPA_2d_to_3d": [
            {
                "name": "2D to 3D with DPA",
                "venv": main_venv,
                "script": os.path.join(script_dir, "src", "evaluation", "run_dpa.py"),
                "cwd": os.path.join(script_dir, "src"),
                "args": ["--config", os.path.join(script_dir, "src", "config.yaml")],
            },
        ],
    }

    # Handle input: if a single list is passed, unpack it; otherwise, treat as multiple strings
    if len(pipeline_types) == 1 and isinstance(pipeline_types[0], list):
        types = pipeline_types[0]  # Unpack the list
    else:
        types = list(pipeline_types)  # Treat as multiple strings

    print(f"Selected pipeline types: {types}")

    # Collect scripts for each type
    scripts = []
    for pipeline_type in types:
        if pipeline_type in pipelines:
            scripts.extend(pipelines[pipeline_type])
        else:
            raise ValueError(f"Unknown pipeline type '{pipeline_type}'. Skipping.")

    return scripts


if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run pipeline scripts.")
    parser.add_argument(
        "--parts",
        "-p",
        nargs="+",
        type=int,
        help="List of parts to run (1-based index)",
    )
    parser.add_argument(
        "-ex", "--exclude", nargs="+", type=int, help="Exclude specific script numbers"
    )

    parser.add_argument(
        "--config",
        default="src/config.yaml",
        type=str,
        help="Path to configuration file (YAML format)",
    )

    args = parser.parse_args()
    # Load config
    config = load_config(args.config)

    parts_to_run = args.parts
    exclude_parts = args.exclude
    device = config.get("device_global", "cuda:0")
    
    # Check if we should use a conda environment instead of venvs
    use_conda_env = config.get("conda_env", "/mnt/lustre/work/lensch/lhr526/.conda/py-310-fresh")
    
    if use_conda_env:
        print(f"ℹ️  Using conda environment: {use_conda_env}")
    else:
        print("ℹ️  Using virtual environments (.venv) for each component")

    # Define regular scripts to run
    scripts_to_run = get_scripts(
        [
            "segmentation",
            "inpainting",
            "camera_dust3r",
            "Hunyuan_2d_to_3d",
            "scene_recon",
            "scene_optim",
            "render",
            "evaluation",
        ]
    )

    # swap out vggt for position 2 of scripts_to_run
    if config.get("Use_VGGT", True):
        scripts_to_run = get_scripts(
            [
                "segmentation",
                "inpainting",
                "Hunyuan_2d_to_3d",
                "camera_vggt",
                "point_cloud_extraction",
                "scene_recon",
                "scene_optim",
                "render",
                "evaluation",
            ]
        )



    if config.get("use_hunyuan21", False):
                scripts_to_run = get_scripts(
            [
                "segmentation",
                "inpainting",
                "Hunyuan_2d_to_3d_v21",
                "camera_vggt",
                "point_cloud_extraction",
                "scene_recon",
                "scene_optim",
                "render",
                "evaluation",
            ]
        )

    if config.get("Use_MIDI", False):
        scripts_to_run = get_scripts(
            [
                "MIDI_2d_to_3d",
                "scene_optim",
                "evaluation",
            ]
        )

    if config.get("Use_DPA", False):
        scripts_to_run = get_scripts(
            [
                "DPA_2d_to_3d",
            ]
        )

    logging.info(f"Scripts to run: {[task['name'] for task in scripts_to_run]}")
    # Start time
    start_time = time.time()

    run_all(
        scripts_to_run,
        parts_to_run,
        exclude_parts,
        start_time=start_time,
        device=device,
        config=config,
        use_conda_env=use_conda_env,
    )

    # End time
    end_time = time.time()
    print(f"Total time taken: {(end_time - start_time)/60} minutes")
