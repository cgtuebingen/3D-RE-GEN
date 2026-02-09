
import argparse, os, logging
from utils.global_utils import (
    load_config,
    create_glb_scene,
    create_pred_ply_scene,
    load_glb_to_point_cloud,
    apply_icp_results_to_glb,
    save_point_cloud,
    match_pointclouds,
)
from scene_optimization.mesh_pointclouds import mesh_background

# from utils.metrics import icp
import shutil

# Suppress PIL decompression bomb warning for large images
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable limit for large scene textures

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
import torch, numpy as np
from sklearn.decomposition import PCA


def align_clouds_pca(pred_points_np, gt_points_np):
    """
    Aligns pred_points_np to the PCA axes and scale of gt_points_np.
    Returns pred_points_np transformed to GT PCA frame and scale.
    """

    # Center both clouds
    gt_centroid = np.mean(gt_points_np, axis=0)
    gt_points_centered = gt_points_np - gt_centroid
    pred_centroid = np.mean(pred_points_np, axis=0)
    pred_points_centered = pred_points_np - pred_centroid

    # PCA axes
    pca_gt = PCA(n_components=3)
    pca_gt.fit(gt_points_centered)
    gt_axes = pca_gt.components_  # (3, 3)

    pca_pred = PCA(n_components=3)
    pca_pred.fit(pred_points_centered)
    pred_axes = pca_pred.components_  # (3, 3)

    # Rotation matrix to align pred_axes to gt_axes
    R = gt_axes.T @ pred_axes

    # Rotate pred points to GT PCA frame
    pred_points_rot = pred_points_centered @ R

    # Scale to GT scale
    gt_scale = np.max(np.linalg.norm(gt_points_centered, axis=1))
    pred_scale = np.max(np.linalg.norm(pred_points_rot, axis=1))
    pred_points_scaled = pred_points_rot / (pred_scale + 1e-8) * gt_scale

    # Optionally, add GT centroid back (if you want to compare in original space)
    # pred_points_aligned = pred_points_scaled + gt_centroid
    # For normalization, keep centered at origin
    return pred_points_scaled


# method to extract all data with marigold
def extract_marigold_data(config):
    import torch
    from diffusers import MarigoldIntrinsicsPipeline, MarigoldNormalsPipeline
    from diffusers.utils import load_image
    from PIL import Image

    # --- Setup ---
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    source_image_path = os.path.join(config["output_inp_banana"], "empty_room.png")
    # check if file exist else error
    if not os.path.exists(source_image_path):
        raise FileNotFoundError(f"Source image not found at {source_image_path}")
    
    image = load_image(source_image_path)
    print(f"Using device: {device}")

    # --- 1. Get Albedo, Roughness, and Metallic ---
    print("Running Appearance Pipeline (Albedo + Roughness + Metallic)...")
    iid_pipe = MarigoldIntrinsicsPipeline.from_pretrained(
        "prs-eth/marigold-iid-appearance-v1-1", variant="fp16", torch_dtype=torch.float16
    ).to(device)

    # Note: The 'intrinsics' variable here contains the raw prediction.
    # We must use the processor to visualize/extract the maps.
    intrinsics_result = iid_pipe(image)

    # Visualize and save the maps
    vis = iid_pipe.image_processor.visualize_intrinsics(
        intrinsics_result.prediction, iid_pipe.target_properties
    )
    # base path
    image_base = config.get("images_marigold_base", "../output/findings/scene_marigold/")
    os.makedirs(image_base, exist_ok=True)

    # vis is a list, so we take the first element
    brdf_maps = vis[0]
    brdf_maps["albedo"].save(os.path.join(image_base, "albedo_map.png"))
    brdf_maps["roughness"].save(os.path.join(image_base, "roughness_map.png"))
    brdf_maps["metallicity"].save(os.path.join(image_base, "metallic_map.png"))  # <-- The new map!
    print("Saved albedo_map.png, roughness_map.png, and metallic_map.png")


    # --- 2. Get Normals ---
    print("Running Normal Pipeline...")
    pipe = MarigoldNormalsPipeline.from_pretrained(
        "prs-eth/marigold-normals-v1-1", variant="fp16", torch_dtype=torch.float16
    ).to("cuda")


    normal_result = pipe(image)# , denoising_steps=10, ensemble_size=5)
    vis = pipe.image_processor.visualize_normals(normal_result.prediction)
    vis[0].save(os.path.join(image_base, "normal_map.png"))
    print("Saved normal_map.png")



def main(config):
    use_midi = config.get("Use_MIDI", False)
    scene_PD = (
        config["glb_scene_path"] if not use_midi else config["glb_scene_path_midi"]
    )

    logging.info(f"Using MIDI: {use_midi}")
    logging.info(f"Predicted scene glb path: {scene_PD}")

    glb_folder = config["glb_output_folder"]

    ply_pred_points = config["ply_pred_points"]
    ply_gt_points = config["ply_gt_points"]

    # Ground truth glb file
    # Extract UUID and scene name
    input_image = config["input_image"]
    scene_path_GT = config.get("GT_scene", None)

    if config["use_3d_front"]:
        _3DFront_scene_dir = config.get("3d_front_scene", "../3D-Front/3D-FRONT-SCENE/")

        uuid = os.path.basename(os.path.dirname(os.path.dirname(input_image)))
        scene_name = os.path.basename(os.path.dirname(input_image))
        # Compose full path to Ground Truth GLB
        scene_path_GT = os.path.join(_3DFront_scene_dir, uuid, f"{scene_name}.glb")

    if scene_path_GT and not os.path.exists(scene_path_GT):
        logging.warning(f"Ground truth scene glb not found: {scene_path_GT}. Skipping GT.")
        scene_path_GT = None

    if scene_path_GT:
        logging.info(f"Ground truth scene glb path: {os.path.abspath(scene_path_GT)}")
    else:
        logging.info("Ground truth scene glb path: None")

    ################################################################################################
    # Extract empty scene from Marigold if specified
    ################################################################################################
    if config.get("use_empty_scene", True) and not use_midi:
        logging.info("Using empty scene extraction from Marigold...")
        # try loading diffusers if not possible skip
        try:
            extract_marigold_data(config)
        except Exception as e:
            logging.error(f"Failed to extract empty scene from Marigold: {e}")

    ################################################################################################
    # Create scene glb from individual object glbs in a folder
    ################################################################################################
    # Save out scene glb
    if use_midi:
        logging.info(
            "Using existing MIDI scene glb and copying it to combined_scene.glb path..."
        )
        # Copy the MIDI generated glb to the expected path for consistency
        if config["glb_scene_path"] != config["glb_scene_path_midi"]:
            shutil.copyfile(config["glb_scene_path_midi"], config["glb_scene_path"])
            logging.info(
                f"Copied MIDI glb from {config['glb_scene_path_midi']} to {config['glb_scene_path']}"
            )

    else:
        logging.info("Creating scene glb from individual object glbs in folder...")
        create_glb_scene(
            input_dir=glb_folder,
            output_path=scene_PD,
            config=config,
            device=config.get("device", "cuda"),  # Default to 'cuda' if not specified
        )

    ################################################################################################
    # Create back-projection scene ply from individual object ply in a folder
    ################################################################################################
    if not use_midi:
        logging.info(
            "Creating back-projection scene ply from individual object plys in folder..."
        )
        create_pred_ply_scene(
            input_dir=config["output_ply"],
            output_path=config["ply_scene_bp_path"],
            device=config.get("device", "cuda"),  # Default to 'cuda' if not specified
        )

    ################################################################################################
    # Create point clouds from created glb scene and GT glb scene
    ################################################################################################

    # Load and process point clouds
    pred_points = load_glb_to_point_cloud(
        glb_path=scene_PD,
        output_file=ply_pred_points,
        num_samples=100000, #config.get("num_samples", 20480),
        skip_textures_material=True,  # Skip textures/materials to avoid PIL issues
        device=config.get("device", "cuda"),
        save=True,
    )
    logging.debug(f"Predicted point cloud shape: {pred_points.shape}")
    # logging.info(f"Predicted pc: {pred_points}")

    gt_points = (
        load_glb_to_point_cloud(
            glb_path=scene_path_GT,
            output_file=ply_gt_points,
            num_samples=100000, # config.get("num_samples", 20480),
            skip_textures_material=True,  # Skip textures/materials to avoid PIL issues
            device=config.get("device", "cuda"),
            save=True,
        )
        if scene_path_GT is not None
        else None
    )

    if gt_points is not None:
        logging.debug(f"Ground truth point cloud shape: {gt_points.shape}")
    else:
        logging.debug("Ground truth point cloud: None")

    ################################################################################################
    # Mesh Background if available
    ############################################################################################

    # mesh background if file available
    pc_folder = config.get("output_vggt", "../output/vggt/sparse")
    pc_file = os.path.join(pc_folder, "points_emptyRoom.ply")

    if os.path.exists(pc_file) and not use_midi:
        logging.debug(f"Loading point cloud from: {pc_file}")
        mesh_background(config, pc_file)
    else:
        logging.warning(f"No background point cloud found at: {pc_file}")

    ################################################################################################
    # Apply ICP alignment if specified
    ################################################################################################

    if gt_points is None:
        logging.warning("Ground truth scene path is None, skipping ICP alignment.")
        return

    logging.debug(
        "Pred points type and shape before ICP:",
        type(pred_points),
        pred_points.shape,
    )
    
    # ============================================================
    # NEW: Normalize both point clouds to unit scale (centered at origin)
    # This ensures fair comparison for distance-based metrics
    # ============================================================
    
    # Convert to numpy if tensor
    if isinstance(gt_points, torch.Tensor):
        gt_points_np = gt_points.squeeze().cpu().numpy() if gt_points.dim() > 2 else gt_points.cpu().numpy()
    else:
        gt_points_np = gt_points
    
    if isinstance(pred_points, torch.Tensor):
        pred_points_np = pred_points.squeeze(0).cpu().numpy() if pred_points.dim() == 3 else pred_points.cpu().numpy()
    else:
        pred_points_np = pred_points
    
    # Normalize GT points
    gt_centroid = np.mean(gt_points_np, axis=0)
    gt_points_centered = gt_points_np - gt_centroid
    gt_scale = np.max(np.linalg.norm(gt_points_centered, axis=1))
    gt_points_normalized = gt_points_centered / gt_scale
    
    logging.info(f"GT normalization: centroid={gt_centroid}, scale={gt_scale}")
    
    # Normalize pred points
    if config.get("use_pca_predPoints_normalization", False):
        pred_points_normalized = align_clouds_pca(pred_points_np, gt_points_normalized)
    else:
        pred_centroid = np.mean(pred_points_np, axis=0)
        pred_points_centered = pred_points_np - pred_centroid
        pred_scale = np.max(np.linalg.norm(pred_points_centered, axis=1))
        pred_points_normalized = pred_points_centered / pred_scale
    
        logging.info(f"Pred normalization: centroid={pred_centroid}, scale={pred_scale}")
    



    # Apply ICP alignment if specified
    if config["use_icp"] or use_midi:
        # Store normalized numpy arrays for later use
        gt_points_np = gt_points_normalized
        pred_points_np = pred_points_normalized
        
        # Convert normalized points to tensors for ICP
        gt_points = (
            torch.from_numpy(gt_points_np)
            .float()
            .unsqueeze(0)
            .to(config.get("device", "cuda"))
        )
        pred_points = (
            torch.from_numpy(pred_points_np)
            .float()
            .unsqueeze(0)
            .to(config.get("device", "cuda"))
        )


        # PyTorch3D exposes iterative_closest_point as a function inside pytorch3d.ops,
        # not a submodule. Import it accordingly, with a safe fallback for older layouts.

        try:
            from pytorch3d.ops import iterative_closest_point as icp
        except Exception:
            import pytorch3d.ops as _p3d_ops

            icp = _p3d_ops.iterative_closest_point

        # Run ICP. PyTorch3D returns an ICPSolution namedtuple with fields like:
        # converged, rmse, Xt, RTs (SimilarityTransform with R, T, s), t_history
        icp_result = icp(
            X=pred_points,
            Y=gt_points,
            max_iterations=config.get("icp_max_iterations", 50),
            estimate_scale=config.get("icp_estimate_scale", False),
            #relative_rmse_thr=1e-6,

            #relative_rmse_thr=config.get("tolerance_icp", 1e-6),
            #estimate_scale=True
        )
      

        final_icp_result = icp_result

        # Extract transformed points and transforms with a robust fallback
        pred_points_aligned = getattr(final_icp_result, "Xt", None)
        RTs = getattr(icp_result, "RTs", None)
        if pred_points_aligned is None:
            # Fallback to tuple indexing if needed
            try:
                pred_points_aligned = icp_result[2]
            except Exception:
                raise RuntimeError(
                    "Unexpected ICP result format: cannot access transformed points (Xt)"
                )


        # Use the aligned GT points from ICP for subsequent steps/saving
        pred_points_normalized = pred_points_aligned



    # Save both normalized and aligned point clouds
    save_point_cloud(gt_points_normalized, ply_gt_points, blender_readable=False)
    
    # Also save the normalized pred points for fair metric comparison
    #ply_pred_normalized = ply_pred_points.replace('.ply', '_normalized.ply')
    save_point_cloud(pred_points_normalized, ply_pred_points, blender_readable=False)
    logging.info(f"Saved normalized pred points to: {ply_pred_points}")



if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Evaluate GLB scenes with MIDI metrics using config"
    )
    parser.add_argument(
        "--config",
        default="../src/config.yaml",
        type=str,
        help="Path to configuration file (YAML format)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Dynamically set logging level based on config['logging']
    level_str = config.get("logging", "INFO").upper()  # Default to 'INFO' if not set
    level = getattr(
        logging, level_str, logging.INFO
    )  # Map string to logging level (e.g., 'DEBUG' -> logging.DEBUG)
    logging.getLogger().setLevel(level)
    logging.info(f"Logging level set to: {level_str}")




    # Run main evaluation
    main(config)
