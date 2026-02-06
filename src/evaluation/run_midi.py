#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_pipeline.py

A pure‑Python entry point that reproduces the Gradio demo for
MIDI‑3D + Grounding‑SAM, but is completely driven by a configuration
file.  It also adds an optional texture stage.

Usage
-----
    python scripts/run_pipeline.py --config config.yaml
"""

from utils.global_utils import load_config

import argparse
import json
import logging
import os
import random
import uuid
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image

# ----------------------------------------------------------------------
# 1️⃣  Imports from your repo
# ----------------------------------------------------------------------
from midi.pipelines.pipeline_midi import MIDIPipeline
from scripts.grounding_sam import (
    detect,
    plot_segmentation,
    prepare_model,
    segment,
)
from scripts.inference_midi import run_midi


# ----------------------------------------------------------------------
# 2️⃣  Helper utilities
# ----------------------------------------------------------------------
def set_random_seed(seed: int) -> None:
    """Make the whole run deterministic (as far as possible)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed}")


def load_image(path: Union[str, Path]) -> Image.Image:
    """Open an image and convert to RGB PIL."""
    img = Image.open(path).convert("RGB")
    logging.info(f"Loaded image {path} (size={img.size})")
    return img


def read_boxes_from_txt(txt_path: Union[str, Path]) -> List[List[int]]:
    """Simple txt format: one line per box, four ints separated by whitespace."""
    boxes = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            boxes.append([int(p) for p in parts])
    logging.info(f"Read {len(boxes)} boxes from {txt_path}")
    return boxes


def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------------
# 3️⃣  Core pipeline functions (mirroring the Gradio callbacks)
# ----------------------------------------------------------------------
@torch.no_grad()
def run_segmentation(
    rgb_image: Image.Image,
    seg_mode: str,
    *,
    boxes: Optional[List[List[int]]] = None,
    text_labels: Optional[str] = None,
    polygon_refinement: bool = True,
    detect_threshold: float = 0.3,
    grounding_models: Tuple[Any, Any, Any],
) -> Image.Image:
    """
    Returns a PIL image visualising the segmentation (the same output that
    Gradio displayed).
    """
    object_detector, sam_processor, sam_segmentator = grounding_models

    segment_kwargs = {}

    if seg_mode == "box":
        if not boxes:
            raise ValueError("Box mode selected but no boxes were supplied.")
        # Gradio used a nested list [[x0, y0, x1, y1], ...]; we keep the same shape.
        segment_kwargs["boxes"] = [
            boxes
        ]  # list‑of‑list because the API expects a batch
    else:  # label mode
        if not text_labels:
            raise ValueError("Label mode selected but `text_labels` is empty.")
        labels = [lbl.strip() for lbl in text_labels.split(",") if lbl.strip()]
        detections = detect(object_detector, rgb_image, labels, detect_threshold)
        segment_kwargs["detection_results"] = detections

    detections = segment(
        sam_processor,
        sam_segmentator,
        rgb_image,
        polygon_refinement=polygon_refinement,
        **segment_kwargs,
    )
    seg_vis = plot_segmentation(rgb_image, detections)
    torch.cuda.empty_cache()

    return seg_vis


@torch.no_grad()
def run_generation(
    pipe: MIDIPipeline,
    rgb_image: Image.Image,
    seg_image: Union[str, Image.Image],
    seed: int,
    num_inference_steps: int = 35,
    guidance_scale: float = 7.0,
    do_image_padding: bool = False,
    tmp_dir: Union[str, Path] = "tmp",
) -> Path:
    """
    Executes the MIDI‑3D pipeline and writes a .glb file to `tmp_dir`.
    Returns the absolute path of the generated GLB.
    """
    with torch.autocast(
        device_type=pipe.device.type, dtype=torch.float16
    ):  # fix for oom TODO
        scene = run_midi(
            pipe,
            rgb_image,
            seg_image,
            seed,
            num_inference_steps,
            guidance_scale,
            do_image_padding,
        )

    out_path = Path(tmp_dir) / f"midi3d_{uuid.uuid4()}.glb"
    scene.export(str(out_path))

    del scene  # free memory

    torch.cuda.empty_cache()
    logging.info(f"Generated GLB saved to {out_path}")
    return out_path


# ----------------------------------------------------------------------
# 4️⃣  Optional texture stage (kept exactly as in the original demo)
# ----------------------------------------------------------------------
def run_texture_stage(
    scene_path: Path,
    rgb_image: Image.Image,
    seg_image: Union[str, Image.Image],
    seed: int,
    tmp_dir: Union[str, Path],
    # The following two pipelines are *only* needed for texturing.
    # They are imported lazily inside the function so that the heavy
    # download only happens when `run_texture` is True.
) -> Path:
    """
    Loads the GLB generated by MIDI‑3D, runs the MV‑Adapter + texture
    pipeline and writes a new GLB with textures.
    """
    # Lazy imports – they pull in large models only when needed.
    from scripts.image_to_textured_scene import (
        prepare_ig2mv_pipeline,
        prepare_texture_pipeline,
        run_i2tex,
    )

    # ------------------------------------------------------------------
    # 4.1 Load the scene
    # ------------------------------------------------------------------
    scene = trimesh.load(str(scene_path), process=False)
    logging.info(f"Loaded scene with {len(scene.geometry)} meshes for texturing")

    # ------------------------------------------------------------------
    # 4.2 Prepare the two texture pipelines (once per run)
    # ------------------------------------------------------------------
    ig2mv_pipe = prepare_ig2mv_pipeline(device="cuda", dtype=torch.float16)
    texture_pipe = prepare_texture_pipeline(device="cuda", dtype=torch.float16)

    # ------------------------------------------------------------------
    # 4.3 Run the texture generation
    # ------------------------------------------------------------------
    out_tmp = Path(tmp_dir) / f"textured_{uuid.uuid4()}"
    out_tmp.mkdir(parents=True, exist_ok=True)

    textured_scene = run_i2tex(
        ig2mv_pipe,
        texture_pipe,
        scene,
        rgb_image,
        seg_image,
        seed,
        output_dir=str(out_tmp),
    )
    out_glb = out_tmp / "textured_scene.glb"
    textured_scene.export(str(out_glb))
    torch.cuda.empty_cache()
    logging.info(f"Textured GLB saved to {out_glb}")
    return out_glb


# ----------------------------------------------------------------------
# 5️⃣  Main driver – reads config, builds everything, runs the pipeline
# ----------------------------------------------------------------------
def main(cfg: dict) -> None:
    # ------------------------------------------------------------------
    # 5.1 Basic sanity & logging
    # ------------------------------------------------------------------
    log_level = logging.DEBUG if cfg.get("debug", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logging.info("=== Starting MIDI‑3D + SAM pipeline ===")

    # ------------------------------------------------------------------
    # 5.2 Device / dtype
    # ------------------------------------------------------------------
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    # dtype = (
    #     torch.bfloat16
    #     if torch.cuda.is_available()
    #     and torch.cuda.get_device_capability()[0] >= 8
    #     else torch.float16
    # )
    dtype = torch.float16
    logging.info(f"Using device={device}, dtype={dtype}")

    # ------------------------------------------------------------------
    # 5.3 Seed handling
    # ------------------------------------------------------------------
    seed = cfg.get("seed", 42)
    if cfg.get("randomize_seed", False):
        seed = random.randint(0, np.iinfo(np.int32).max)
        logging.info(f"Randomized seed -> {seed}")
    set_random_seed(seed)

    # ------------------------------------------------------------------
    # 5.4 Paths
    # ------------------------------------------------------------------
    img_path = Path(cfg["image_url"])
    output_dir = ensure_dir(cfg.get("midi_output", "outputs"))
    tmp_dir = ensure_dir(cfg.get("midi_tmp", "tmp"))

    # ------------------------------------------------------------------
    # 5.5 Load the input image (RGB)
    # ------------------------------------------------------------------
    rgb_image = load_image(img_path)
    # # Optionally resize to a max size
    # max_size = cfg.get("max_image_size_midi", 512)

    # if max_size and max(rgb_image.size) > max_size:
    #     scale = max_size / max(rgb_image.size)
    #     new_w = int(rgb_image.width * scale)
    #     new_h = int(rgb_image.height * scale)
    #     rgb_image = rgb_image.resize((new_w, new_h), Image.LANCZOS)
    #     logging.info(f"Resized input image to {rgb_image.size}")

    # ------------------------------------------------------------------
    # 5.6 ----------------------------------------------------------------
    #   1️⃣  Prepare Grounding‑SAM models
    # ------------------------------------------------------------------
    object_detector, sam_processor, sam_segmentator = prepare_model(
        device=device,
        detector_id="IDEA-Research/grounding-dino-tiny",
        segmenter_id="facebook/sam-vit-base",
    )
    grounding_models = (object_detector, sam_processor, sam_segmentator)

    # clean cache from segmentation models
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    #   2️⃣  Prepare MIDI‑3D pipeline
    # ------------------------------------------------------------------
    midi_repo = "VAST-AI/MIDI-3D"
    midi_weights_dir = Path("pretrained_weights/MIDI-3D")

    if not midi_weights_dir.is_dir():
        logging.info(f"Downloading MIDI‑3D weights to {midi_weights_dir}")
        snapshot_download(repo_id=midi_repo, local_dir=str(midi_weights_dir))
    pipe: MIDIPipeline = MIDIPipeline.from_pretrained(str(midi_weights_dir)).to(
        device, dtype
    )
    pipe.init_custom_adapter(
        set_self_attn_module_names=[
            "blocks.8",
            "blocks.9",
            "blocks.10",
            "blocks.11",
            "blocks.12",
        ]
    )
    logging.info("MIDI‑3D pipeline ready")

    # ------------------------------------------------------------------
    #   3️⃣  Segmentation
    # ------------------------------------------------------------------
    seg_mode = cfg.get("seg_mode", "label")
    boxes = None
    if seg_mode == "box":
        # Prefer a literal list in the yaml, otherwise read a txt file.
        if "boxes" in cfg:
            boxes = cfg["boxes"]
        elif "boxes_path" in cfg:
            boxes = read_boxes_from_txt(cfg["boxes_path"])
        else:
            raise ValueError(
                "Box mode selected but no `boxes` or `boxes_path` provided."
            )

    text_labels = "furniture, lamps, decoration" #cfg.get("labels", "furniture, lights")
    # transform list of labels to comma-separated string
    if isinstance(text_labels, list):
        text_labels = ", ".join(text_labels)
    text_labels = text_labels.strip()

    # raise error when no labels found, else print them
    if seg_mode == "label" and not text_labels:
        raise ValueError("Label mode selected but no `labels` provided.")
    else:
        logging.info(f"Using text labels: {text_labels}")

    seg_image = run_segmentation(
        rgb_image,
        seg_mode,
        boxes=boxes,
        text_labels=text_labels,
        polygon_refinement=True,
        detect_threshold=cfg.get("detect_threshold", 0.3),
        grounding_models=grounding_models,
    )

    # Save the segmentation visualisation (useful for debugging)
    seg_vis_path = output_dir / "seg_vis_seg_midi.png"
    seg_image.save(seg_vis_path)
    logging.info(f"Segmentation visualisation saved to {seg_vis_path}")

    # ------------------------------------------------------------------
    #   4️⃣  Generation (MIDI‑3D)
    # ------------------------------------------------------------------
    final_glb = cfg["glb_scene_path_midi"]
    print("Saving final glb to ", os.path.abspath(final_glb))

    if not cfg.get("use_latest_glb", False):
        try:
            glb_path = run_generation(
                pipe,
                rgb_image,
                seg_image,
                seed=seed,
                num_inference_steps=cfg.get("num_inference_steps_midi", 35),
                guidance_scale=cfg.get("guidance_scale_midi", 7.0),
                do_image_padding=cfg.get("do_image_padding", False),
                tmp_dir=tmp_dir,
            )

        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.error("CUDA OOM – exiting gracefully.")
                torch.cuda.empty_cache()
                return
            else:
                raise

        logging.info(f"MIDI‑3D generation completed, GLB at {glb_path}")

        # Move the generated GLB to the final output folder
        glb_path.rename(final_glb)
        logging.info(f"Final GLB written to {final_glb}")

    # ------------------------------------------------------------------
    #   5️⃣  Optional texture stage
    # ------------------------------------------------------------------
    if cfg.get("run_texture", False):
        tex_seed = (seed,)
        textured_glb = run_texture_stage(
            final_glb,
            rgb_image,
            seg_image,
            seed=tex_seed,
            tmp_dir=tmp_dir,
        )
        final_tex_glb = output_dir / f"scene_textured_{uuid.uuid4()}.glb"
        textured_glb.rename(final_tex_glb)
        logging.info(f"Final textured GLB written to {final_tex_glb}")

    logging.info("=== Pipeline finished successfully ===")


# ----------------------------------------------------------------------
# 6️⃣  CLI entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MIDI‑3D + SAM pipeline from a config file."
    )
    parser.add_argument(
        "--config",
        default="../src/config.yaml",
        type=str,
        help="Path to configuration file (YAML format)",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    main(cfg)
