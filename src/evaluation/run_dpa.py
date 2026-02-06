import subprocess
import os

def run_script(venv_path, script_path, cwd, *args):
    # Convert to absolute paths
    # base_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # venv_path = os.path.abspath(os.path.join(base_folder, venv_path))
    # script_path = os.path.abspath(os.path.join(base_folder, script_path))
    # cwd = os.path.abspath(os.path.join(base_folder, cwd))
    print(f"Using virtual environment at: {os.path.abspath(venv_path)}")
    print(f"Running script: {os.path.abspath(script_path)} in cwd: {os.path.abspath(cwd)} with args: {args}")

    # Construct the path to the Python executable within the virtual environment

    python_executable = os.path.join(venv_path, "bin", "python3.12")
    print("Absolute path to Python executable:", python_executable)
    command = [python_executable, script_path, *args]
    subprocess.run(command, cwd=cwd, check=True)

def main():
    output_path = "../outputs/dpa"
    input_path = "..input_images/scene"
    image_id = "interioer_design_1.jpg" #"479d2d66-4d1a-47ca-a023-4286fc547301---rgb_0017"
    gpu_id = "0"
    base_folder = "DeepPriorAssembly"

    # print cwd
    print("Current working directory:", os.getcwd())

    # Segmentation
    run_script(f"../{base_folder}/grounded_sam/.venv", f"../{base_folder}/grounded_sam/segment_scenes.py", f"../{base_folder}/grounded_sam",
               "--input_path", input_path, "--save_path", f"{output_path}/segmentation", "--image_id", image_id)

    # Inpainting
    run_script("stablediffusion/.venv", "stablediffusion/scripts/img2img_inpainting.py", "stablediffusion",
               "--input_path", f"{output_path}/segmentation", "--outdir", f"{output_path}/inpainting",
               "--n_samples", "6", "--strength", "0.5", "--image_id", image_id,
               "--ckpt", "stablediffusion/checkpoints/v2-1_512-ema-pruned.ckpt")

    # Object Generation
    run_script("shap-e/.venv", "shap-e/object_generation.py", "shap-e",
               "--input_path", f"{output_path}/inpainting", "--output_path", f"{output_path}/object_generation",
               "--image_id", image_id)

    # Geometry Estimation
    run_script("dust3r/.venv", "dust3r/gen_scene_geometry.py", "dust3r",
               "--input_path", input_path, "--output_path", output_path, "--image_id", image_id)

    # Final Registration by Optimization
    run_script("dust3r/.venv", "registration/optimization_5dof.py", "registration",
               "--image_id", image_id, "--geometry_dir", f"{output_path}/geometry",
               "--mask_dir", f"{output_path}/segmentation", "--object_dir", f"{output_path}/object_generation",
               "--output_dir", f"{output_path}/final_registration")

if __name__ == "__main__":
    main()