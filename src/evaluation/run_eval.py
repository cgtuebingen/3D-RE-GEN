import os
import yaml
import argparse
import trimesh
from PIL import Image

import lpips

import torch
# from pytorch3d.io import load_objs_as_meshes

from utils.metrics import (
    # compute_chamfer_distance,
    compute_fscore,
    compute_volume_iou,
    # icp,
)  # From your metrics.py
from pytorch3d.loss import chamfer_distance as pcd_chamfer_distance

import point_cloud_utils as pcu
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

import numpy as np

# import ssim and psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


from utils.global_utils import load_config, create_glb_scene, save_point_cloud
from utils.eval_utils import (
    dump_evaluation,
    load_metrics,
    get_previous_evaluation,
    compare_metrics_to_csv,
)
import csv

# Set up logging
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def compute_precision_recall(pred_points, gt_points, threshold=0.01):
    """
    Computes precision and recall between predicted and ground truth point clouds.
    Args:
        pred_points (np.ndarray): Predicted point cloud of shape (N, 3).
        gt_points (np.ndarray): Ground truth point cloud of shape (M, 3).
        threshold (float): Distance threshold for considering a point as a match.
    Returns:
        precision (float): Precision of the predicted points.
        recall (float): Recall of the predicted points.
    """
    pred_tree = cKDTree(pred_points)
    gt_tree = cKDTree(gt_points)

    distances, _ = pred_tree.query(gt_points)
    precision = (distances < threshold).mean()

    distances, _ = gt_tree.query(pred_points)
    recall = (distances < threshold).mean()

    return precision, recall


def main(config):
    """Main evaluation function using config parameters"""
    # Setup temporary directory
    tmp_path = config.get("tmp_dir", os.path.join(os.getcwd(), "output"))
    os.makedirs(tmp_path, exist_ok=True)

    device = config.get("device", "cuda:0")
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # initialize LPIPS model
    lp_model = lpips.LPIPS(net="alex").to(device)

    # Import and set iamges
    gt_image = config["image_url"]
    if not os.path.exists(gt_image):
        raise FileNotFoundError(f"Ground truth image not found at {gt_image}")

    # import image path as iamge then numpy array
    gt_image = Image.open(gt_image).convert("RGB")
    # Convert to numpy array
    gt_image = np.array(gt_image)

    pd_image = config.get("predicted_image", None)
    if pd_image and not os.path.exists(pd_image):
        raise FileNotFoundError(f"Predicted image not found at {pd_image}")

    # import image path as image then numpy array
    pd_image = Image.open(pd_image).convert("RGB")
    # image to numpy array
    pd_image = np.array(pd_image)

    # normalize images to -1 to 1 range for ssim and lpips
    gt_image = (gt_image / 255.0) * 2 - 1
    pd_image = (pd_image / 255.0) * 2 - 1

    pp_path = config["ply_pred_points"]
    gtp_path = config["ply_gt_points"]

    # Load predicted points from PLY file
    if not os.path.exists(pp_path):
        raise FileNotFoundError(f"Predicted PLY file not found at {pp_path}")
    pred_mesh = trimesh.load(pp_path)
    pred_points_np = pred_mesh.vertices  # Shape: (N, 3) numpy array
    pred_points = (
        torch.from_numpy(pred_points_np).unsqueeze(0).to(device, dtype=torch.float32)
    )  # Shape: (1, N, 3)

    # Load ground truth points from PLY file
    if not os.path.exists(gtp_path):
        raise FileNotFoundError(f"Ground truth PLY file not found at {gtp_path}")
    gt_mesh = trimesh.load(gtp_path)
    gt_points_np = gt_mesh.vertices  # Shape: (N, 3) numpy array
    gt_points = (
        torch.from_numpy(gt_points_np).unsqueeze(0).to(device, dtype=torch.float32)
    )  # Shape: (1, N, 3)

    ##################################################################################################
    # 3D image metrics
    ##################################################################################################
    #chamfer, cd2, cd3 = compute_chamfer_distance(
    #    pred_points, gt_points, device=device
    #)  # chamfer distance from MIDI
    fscore = compute_fscore(pred_points, gt_points)
    iou_bbox = compute_volume_iou(pred_points, gt_points, mode="bbox")

    # chamfer distance from point-cloud-utils
    # torch tensor to numpy
    pred_points_np = pred_points.squeeze(0).cpu().numpy().astype(np.float32)
    gt_points_np = gt_points.squeeze(0).cpu().numpy().astype(np.float32)

    # fix for icp issue
    pred_points_np = np.ascontiguousarray(pred_points_np)
    gt_points_np = np.ascontiguousarray(gt_points_np)

    # print shape and some details to check functionality
    logging.debug(
        f"Predicted points shape: {pred_points_np.shape}, dtype: {pred_points_np.dtype}"
    )
    logging.debug(
        f"Ground truth points shape: {gt_points_np.shape}, dtype: {gt_points_np.dtype}"
    )
    # compute chamfer distance using point-cloud-utils
    pcu_cd = pcu.chamfer_distance(
        pred_points_np, gt_points_np, p_norm=2, max_points_per_leaf=20
    )
    # compute Hausdorff distance using point-cloud-utils
    pcu_hd = pcu.hausdorff_distance(pred_points_np, gt_points_np)

    ptd_cd, _ = pcd_chamfer_distance(
        pred_points, gt_points, point_reduction="mean", norm=2
    )

    precision, recall = compute_precision_recall(
        pred_points_np, gt_points_np, threshold=0.01
    )

    # Compute Wasserstein Distance
    wd = wasserstein_distance(pred_points_np.flatten(), gt_points_np.flatten())

    ##################################################################################################
    # 2D image metrics
    ##################################################################################################

    psnr_value = psnr(gt_image, pd_image, data_range=1)
    ssim_value = ssim(
        im1=gt_image,
        im2=pd_image,
        win_size=None,
        gradient=False,
        data_range=1,
        channel_axis=2,
        full=False,
    )

    # GT and PD images to tensor for LPIPS
    gt_image_tensor = (
        torch.tensor(gt_image, dtype=torch.float32, device=device)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )  # Shape: (1, C, H, W)
    pd_image_tensor = (
        torch.tensor(pd_image, dtype=torch.float32, device=device)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )  # Shape: (1, C, H, W)

    lpips_value = lp_model.forward(gt_image_tensor, pd_image_tensor)

    metrics = {
        # point‑cloud‑utils
        #"pcu_chamfer": float(pcu_cd),

        # pytorch3d
        "CD": float(ptd_cd.item()),
        "FSCORE": fscore.item(),
        "IOU_BBOX": iou_bbox.item(),
        "HAUSDORFF": float(pcu_hd),
        
        # midi metrics (example – adapt the names you actually use)
        #"midi_chamfer": chamfer.item(),
        #"midi_cd2": cd2.item(),
        #"midi_cd3": cd3.item(),
        # extra
        "WASSERSTEIN": wd,
        "PRECISION": precision,
        "RECALL": recall,

        # 2‑D image metrics
        "PSNR (10^-2)": psnr_value * 0.01,  # Scale PSNR to be between 0 and 1
        "SSIM": ssim_value,
        "LPIPS": lpips_value.item(),
    }

    # Pretty print metrics
    logging.info("Evaluation Metrics:")
    for key, value in metrics.items():
        logging.info(f"  {key}: {value:.6f}")

    # dump evaluation results
    out_dir = dump_evaluation(
        metrics, config, out_root=config.get("eval_output_dir", "output/evaluation")
    )

    # `out_dir` is the folder we just created with `dump_evaluation`
    prev_dir = get_previous_evaluation(out_dir)

    # current LaTeX table
    cur_metrics = load_metrics(out_dir)
    cur_csv_path = out_dir / "metrics.csv"
    with cur_csv_path.open("w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in cur_metrics.items():
            writer.writerow([k.replace('_', ' ').title(), f"{v:.6f}"])

    # 2️⃣  optional comparison table
    if prev_dir:
        prev_metrics = load_metrics(prev_dir)
        compare_metrics_to_csv(prev_metrics, cur_metrics,
                            prev_name=prev_dir.name,
                            cur_name=out_dir.name,
                            csv_path=out_dir / "comparison.csv")
    else:
        print("No previous evaluation found for comparison.")


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
