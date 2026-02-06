import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from segment_anything import SamPredictor, build_sam
from transformers import AutoProcessor

# We need the dataclasses from the shared data types module
from utils.data_types import DetectionResult, BoundingBox
import torch
from PIL import Image

# Global variable to store the final state when the user is done
final_detections_state = None

# Global variables to hold the segmentation model and processor
segmentator_predictor = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import os

# ---------------------------------------------------------------------------
# Performance caches (font & colors) to avoid recreating heavy objects
# ---------------------------------------------------------------------------
try:
    FONT_CACHE = ImageFont.truetype("arial.ttf", 24)
except IOError:
    FONT_CACHE = ImageFont.load_default()

DISTINCT_COLORS = np.array([
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (255, 192, 203),  # Pink
    (0, 128, 128),    # Teal
    (128, 128, 0),    # Olive
    (255, 165, 0),    # Dark Orange
    (64, 224, 208),   # Turquoise
    (255, 20, 147),   # Deep Pink
    (138, 43, 226),   # Blue Violet
    (34, 139, 34),    # Forest Green
    (220, 20, 60),    # Crimson
    (0, 206, 209),    # Dark Turquoise
    (184, 134, 11),   # Dark Goldenrod
    (139, 69, 19),    # Saddle Brown
    (255, 69, 0),     # Orange Red
    (32, 178, 170),   # Light Sea Green
    (218, 165, 32),   # Goldenrod
    (199, 21, 133),   # Medium Violet Red
    (72, 209, 204),   # Medium Turquoise
    (173, 255, 47),   # Green Yellow
    (255, 215, 0),    # Gold
    (240, 128, 128),  # Light Coral
    (144, 238, 144),  # Light Green
    (175, 238, 238),  # Pale Turquoise
], dtype=np.uint8)

# Track whether we've already set the SAM image this session
_sam_image_initialized = False

def load_segmentation_model(config: dict):
    """Loads the SAM predictor into a global variable."""
    global segmentator_predictor
    
    if segmentator_predictor is None:
        try:
            sam_checkpoint = config.get("segmenter_checkpoint", "sam_vit_h_4b8939.pth")
            
            # Check if the checkpoint file exists
            if not os.path.exists(sam_checkpoint):
                print("---" * 20)
                print("SAM model checkpoint not found!")
                print(f"Please download it by running the following command in your terminal:")
                print(f"wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
                print("---" * 20)
                raise FileNotFoundError(f"SAM checkpoint not found at {sam_checkpoint}")

            print(f"Loading SAM model from {sam_checkpoint}")
            sam = build_sam(checkpoint=sam_checkpoint)
            sam.to(device=device)
            segmentator_predictor = SamPredictor(sam)
            print("SAM predictor loaded successfully.")
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            segmentator_predictor = None


def segment_with_points(
    image_array: np.ndarray,
    detection_dicts: list,
    points_dict: dict,
    config: dict = None,
    segment_function = None
):
    """
    Run segmentation using points to refine existing detections or create new ones.
    """
    # Ensure the model is loaded before starting segmentation
    load_segmentation_model(config)
    
    if segment_function is None:
        raise ValueError("segment_function is required for point-based segmentation")
    
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image_array)
    
    # Prepare points in SAM format
    all_points = []
    all_labels = []
    
    # Add positive points (label=1)
    for point in points_dict.get("positive", []):
        all_points.append([point[0], point[1]])
        all_labels.append(1)
    
    # Add negative points (label=0) 
    for point in points_dict.get("negative", []):
        all_points.append([point[0], point[1]])
        all_labels.append(0)
    
    print(f"Debug: Processing {len(all_points)} points: {all_points}")
    print(f"Debug: Point labels: {all_labels}")
    
    if not all_points:
        print("Debug: No points provided, returning empty results")
        return []
    
    # For each detection, run segmentation with the points
    results = []
    for i, detection_dict in enumerate(detection_dicts):
        try:
            # Get existing refinement context if available
            existing_logits = detection_dict.get("logits", None)
            existing_mask = detection_dict.get("mask", None)
            
            # Create a custom segment call with points
            detection_with_points = segment_single_with_points(
                pil_image,
                detection_dict,
                all_points,
                all_labels,
                config,
                existing_logits,
                existing_mask,
            )
            if detection_with_points:
                results.append(detection_with_points)
        except Exception as e:
            print(f"Error segmenting with points: {e}")
            continue
    
    return results


def segment_single_with_points(
    image: Image.Image,
    detection_dict: dict,
    points: list,
    point_labels: list,
    config: dict = None,
    existing_logits: np.ndarray = None,
    existing_mask: np.ndarray = None,
):
    """
    Segment a single detection using points, with better mask preservation.
    """
    global segmentator_predictor
    
    # Ensure the model is loaded
    if segmentator_predictor is None:
        print("Error: Segmentation model not loaded. Call load_segmentation_model first.")
        return None

    try:
        # Image is assumed already set once at session start for this static editor.

        # Prepare input - use a tighter box to avoid distant parts
        box = detection_dict["box"]
        input_box = np.array([
            box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        ], dtype=np.float32)
        
        # Debug: Show what points we received
        print(f"Debug segment_single_with_points: points={points}, labels={point_labels}")
        
        # Format inputs for SAM
        input_points = np.array(points, dtype=np.float32)
        input_labels = np.array(point_labels, dtype=np.int32)

        # Prepare mask_input from existing logits with correct shape/dtype
        mask_input_np = None
        if existing_logits is not None:
            if isinstance(existing_logits, torch.Tensor):
                existing_logits = existing_logits.detach().cpu().numpy()
            existing_logits = np.array(existing_logits)
            if existing_logits.ndim == 2:
                existing_logits = existing_logits[None, :, :]
            elif existing_logits.ndim == 3 and existing_logits.shape[0] != 1:
                existing_logits = existing_logits[:1, :, :]
            mask_input_np = existing_logits.astype(np.float32)
        
        # Run inference (mask_input expects low-res logits with a batch dim)
        masks, scores, logits = segmentator_predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_box,
            mask_input=mask_input_np,
            multimask_output=False,
        )
        
        # Convert to detection result
        if len(masks) > 0:
            new_mask = masks[0]
            # If adding positive points, enforce additive behavior with previous mask
            # Only union when user provided only positive points in this refinement
            if existing_mask is not None and np.all(input_labels == 1):
                # Ensure shapes match before union
                if existing_mask.shape != new_mask.shape:
                    # Resize existing_mask to prediction resolution if needed (should normally match)
                    existing_mask_resized = cv2.resize(
                        existing_mask.astype(np.uint8),
                        (new_mask.shape[1], new_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                else:
                    existing_mask_resized = existing_mask
                new_mask = np.logical_or(new_mask, existing_mask_resized)

            # Create bounding box from final mask
            if np.any(new_mask):
                y_coords, x_coords = np.where(new_mask)
                new_box = BoundingBox(
                    xmin=int(np.min(x_coords)),
                    ymin=int(np.min(y_coords)),
                    xmax=int(np.max(x_coords)),
                    ymax=int(np.max(y_coords))
                )
                
                return DetectionResult(
                    score=scores[0],
                    label=detection_dict["label"],
                    box=new_box,
                    mask=new_mask,
                    logits=logits[0]  # Store the returned logits for the next iteration
                )
        
        return None
        
    except Exception as e:
        print(f"Error in segment_single_with_points: {e}")
        return None

def render_image_with_masks(base_image: np.ndarray, detections: list[DetectionResult], map_box_fn=None, show_labels: bool = True, show_bboxes: bool = True, bbox_thickness: int = 3) -> Image.Image:
    """Fast, vectorized mask + bbox + label overlay.
    map_box_fn: optional callable (xmin,ymin,xmax,ymax)->(dxmin,dymin,dxmax,dymax) for coordinate scaling (e.g. full->display).
    show_labels: If True, draws labels on masks.
    show_bboxes: If True, draws bounding boxes around masks.
    bbox_thickness: The thickness of the bounding box lines.
    """
    pil_image = Image.fromarray(base_image)
    W, H = pil_image.size

    overlay = np.zeros((H, W, 4), dtype=np.uint8)
    alpha_val = 100

    for i, det in enumerate(detections):
        if det is None or det.mask is None:
            continue
        mask = det.mask
        # Normalize mask shape: expect (H,W) boolean
        if mask.ndim > 2:
            # Common issues: (H,W,1) or extraneous channel dims
            mask = mask.squeeze()
        if mask.dtype != bool:
            mask = mask.astype(bool)
        # Resize mask to display size if needed
        if mask.shape[0] != H or mask.shape[1] != W:
            try:
                mask_resized = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            except Exception:
                continue
            mask = mask_resized
        color = DISTINCT_COLORS[i % len(DISTINCT_COLORS)]
        overlay[mask, 0:3] = color
        overlay[mask, 3] = alpha_val

    base_rgba = pil_image.convert("RGBA")
    overlay_img = Image.fromarray(overlay, mode="RGBA")
    result_image = Image.alpha_composite(base_rgba, overlay_img)

    draw = ImageDraw.Draw(result_image)
    font = FONT_CACHE
    for i, det in enumerate(detections):
        if det is None or det.mask is None:
            continue
        color = tuple(int(c) for c in DISTINCT_COLORS[i % len(DISTINCT_COLORS)])
        xmin, ymin, xmax, ymax = det.box.xmin, det.box.ymin, det.box.xmax, det.box.ymax
        if map_box_fn is not None:
            xmin, ymin, xmax, ymax = map_box_fn(xmin, ymin, xmax, ymax)
        
        if show_bboxes:
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=bbox_thickness)
        
        if show_labels:
            label_text = f"{i}: {det.label}"
            # Center label in mapped coordinates
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2
            try:
                text_w, text_h = draw.textbbox((0, 0), label_text, font=font)[2:]
            except Exception:
                try:
                    text_w, text_h = draw.textsize(label_text, font=font)
                except Exception:
                    text_w = len(label_text) * 10
                    text_h = 14
            tx = cx - text_w // 2
            ty = cy - text_h // 2
            # Outline
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx or dy:
                        draw.text((tx + dx, ty + dy), label_text, fill=(0, 0, 0), font=font)
            draw.text((tx, ty), label_text, fill=color, font=font)

    return result_image.convert("RGB")

def get_detection_choices(detections: list[DetectionResult]) -> list[str]:
    """Creates a list of strings for the checkbox group."""
    return [f"{i}: {det.label}" for i, det in enumerate(detections)]

def render_image_with_points(base_image: np.ndarray, detections: list[DetectionResult], points_dict: dict, map_full_to_display_fn=None, map_box_fn=None) -> Image.Image:
    """Draws masks, labels, and modification points on the (possibly downscaled) image.
    points_dict is assumed to store FULL-resolution coordinates; supply map_full_to_display_fn if base_image is display-sized.
    """
    image = render_image_with_masks(base_image, detections, map_box_fn=map_box_fn)
    draw = ImageDraw.Draw(image)
    def map_pt(pt):
        if map_full_to_display_fn is None:
            return pt
        return map_full_to_display_fn(pt[0], pt[1])
    # Positive points
    for x_full, y_full in points_dict.get("positive", []):
        x, y = map_pt((x_full, y_full))
        draw.ellipse([x-5, y-5, x+5, y+5], fill=(0,255,0), outline=(0,150,0), width=2)
    # Negative points
    for x_full, y_full in points_dict.get("negative", []):
        x, y = map_pt((x_full, y_full))
        draw.ellipse([x-5, y-5, x+5, y+5], fill=(255,0,0), outline=(150,0,0), width=2)
    return image

def render_image_with_bbox_preview(base_image: np.ndarray, detections: list[DetectionResult], bbox_points_full: list, map_full_to_display_fn=None, map_box_fn=None) -> Image.Image:
    """Draws masks, labels, and bounding box preview.
    bbox_points_full: list of full-resolution points (0..2 entries)
    """
    image = render_image_with_masks(base_image, detections, map_box_fn=map_box_fn)
    draw = ImageDraw.Draw(image)
    def map_pt(pt):
        if map_full_to_display_fn is None:
            return pt
        return map_full_to_display_fn(pt[0], pt[1])
    if len(bbox_points_full) >= 1:
        x1, y1 = map_pt(bbox_points_full[0])
        draw.ellipse([x1-6, y1-6, x1+6, y1+6], fill=(255,255,0), outline=(200,200,0), width=3)
        if len(bbox_points_full) >= 2:
            x2, y2 = map_pt(bbox_points_full[1])
            draw.ellipse([x2-6, y2-6, x2+6, y2+6], fill=(255,255,0), outline=(200,200,0), width=3)
            xmin, xmax = min(x1,x2), max(x1,x2)
            ymin, ymax = min(y1,y2), max(y1,y2)
            for offset in range(0,4,2):
                draw.rectangle([xmin-offset, ymin-offset, xmax+offset, ymax+offset], outline=(255,255,0,200), width=2)
            corner_size = 10
            for corner_x, corner_y in [(xmin,ymin),(xmax,ymin),(xmin,ymax),(xmax,ymax)]:
                draw.rectangle([corner_x-corner_size//2, corner_y-corner_size//2, corner_x+corner_size//2, corner_y+corner_size//2], fill=(255,255,0), outline=(200,200,0), width=2)
    return image

def edit_segmentations_interactive(
    image_array: np.ndarray, 
    initial_detections: list[DetectionResult],
    config: dict = None,
    segment_function = None
):
    """
    Launches a Gradio interface to manually edit segmentations with live segmentation updates.
    Returns the final list of DetectionResult objects.
    """
    global final_detections_state
    final_detections_state = initial_detections.copy()
    session_finished = {"finished": False}

    # ---------------- Display Downscaling / Coordinate Mapping --------------
    # Fit the image into a max box while preserving aspect ratio to avoid squashing
    max_display_w, max_display_h = 800, 600
    full_h, full_w = image_array.shape[0], image_array.shape[1]
    # Compute uniform scale so that the image fits inside the box without distortion
    scale = min(max_display_w / full_w, max_display_h / full_h)
    display_w = int(round(full_w * scale))
    display_h = int(round(full_h * scale))

    if (full_w, full_h) != (display_w, display_h):
        try:
            display_image_array = cv2.resize(
                image_array, (display_w, display_h), interpolation=cv2.INTER_AREA
            )
        except Exception:
            display_image_array = image_array
            display_w, display_h = full_w, full_h
    else:
        display_image_array = image_array

    scale_x_display_to_full = full_w / display_w
    scale_y_display_to_full = full_h / display_h
    scale_x_full_to_display = display_w / full_w
    scale_y_full_to_display = display_h / full_h

    def map_display_to_full(xd: int, yd: int):
        return int(round(xd * scale_x_display_to_full)), int(round(yd * scale_y_display_to_full))

    def map_full_to_display(xf: int, yf: int):
        return int(round(xf * scale_x_full_to_display)), int(round(yf * scale_y_full_to_display))

    def map_box_full_to_display(xmin, ymin, xmax, ymax):
        x1, y1 = map_full_to_display(xmin, ymin)
        x2, y2 = map_full_to_display(xmax, ymax)
        return x1, y1, x2, y2

    # Shared display settings that persist across function calls
    display_settings = {"show_labels": True, "show_bboxes": True, "bbox_thickness": 3}

    def ui_render_display(detections, show_labels=None, show_bboxes=None, bbox_thickness=None):
        """Render detections on the display image with proper box/label scaling."""
        # Use provided values or fall back to shared settings
        sl = show_labels if show_labels is not None else display_settings["show_labels"]
        sb = show_bboxes if show_bboxes is not None else display_settings["show_bboxes"]
        bt = bbox_thickness if bbox_thickness is not None else display_settings["bbox_thickness"]
        return render_image_with_masks(
            display_image_array,
            detections,
            map_box_fn=map_box_full_to_display,
            show_labels=sl,
            show_bboxes=sb,
            bbox_thickness=bt
        )

    # Initialize SAM predictor & set image once (performance optimization)
    global _sam_image_initialized
    try:
        load_segmentation_model(config)
        if segmentator_predictor is not None and not _sam_image_initialized:
            segmentator_predictor.set_image(np.array(Image.fromarray(image_array)))
            _sam_image_initialized = True
            print("SAM image set once at session start.")
    except Exception as e:
        print(f"Warning: could not set SAM image at session start: {e}")
    
    # Store points for modification (positive and negative)
    modification_points = {"positive": [], "negative": []}
    bbox_points = []  # Store points for bounding box creation

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        # Use gr.State to hold the list of detections
        detections_state = gr.State(value=initial_detections)
        show_labels_state = gr.State(value=True)
        show_bboxes_state = gr.State(value=True)
        bbox_thickness_state = gr.State(value=3)

        gr.Markdown("# Interactive Segmentation Editor")
        gr.Markdown("""
        **Instructions:**
        - **Delete**: Select masks from the list and click 'Delete Selected'
        - **Modify**: Select masks, click on image to add positive/negative points, then 'Update Masks'
        - **Add New**: Enter label, click points on image, then 'Update Masks' to create new mask
        - **Iterate**: Keep adding points and updating until satisfied
        - **Finish**: Click 'Finish Session' to save final results
        """)
        

        with gr.Row():
            with gr.Column(scale=3):
                # The main image display for clicking points
                image_display = gr.Image(
                    value=ui_render_display(initial_detections),
                    label="Click two points to define opposite corners of a bounding box.",
                    interactive=True,
                    height=display_h,
                    width=display_w,
                )
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Current Masks")
                detection_checkboxes = gr.CheckboxGroup(
                    choices=get_detection_choices(initial_detections),
                    label="Select masks to delete or modify",
                )
                
                delete_button = gr.Button("Delete Selected", size="sm", variant="secondary")

                gr.Markdown("---")
                gr.Markdown("### Edit Mode")
                edit_mode = gr.Radio(
                    choices=["Points", "Bounding Box"],
                    label="Editing method",
                    value="Bounding Box"
                )
                
                gr.Markdown("### Point Mode")
                point_mode = gr.Radio(
                    choices=["Positive (include)", "Negative (exclude)"],
                    label="Click mode",
                    value="Positive (include)",
                    visible=False  # Will be controlled by edit_mode
                )
                
                # Points interface (initially hidden)
                points_group = gr.Group(visible=False)
                with points_group:
                    gr.Markdown("### Modification Points")
                    points_display = gr.Textbox(
                        label="Current points",
                        value="No points added yet",
                        interactive=False
                    )
                    
                    with gr.Row():
                        clear_points_button = gr.Button("Clear Points", size="sm")
                        undo_last_point_button = gr.Button("Undo Last", size="sm")
                    
                    update_masks_button = gr.Button("Update Masks", size="sm", variant="primary")
                
                # Bounding box interface (default visible)  
                bbox_group = gr.Group(visible=True)
                with bbox_group:
                    gr.Markdown("### Bounding Box Editor")
                    gr.Markdown("Click and drag on the main image to draw a bounding box")
                    bbox_points_display = gr.Textbox(
                        label="Bounding box status",
                        value="No bounding box drawn",
                        interactive=False
                    )
                    with gr.Row():
                        clear_bbox_button = gr.Button("Clear Box", size="sm")
                        apply_bbox_button = gr.Button("Apply Box", size="sm", variant="primary")

                gr.Markdown("---")
                gr.Markdown("### Add New Mask")
                new_label_textbox = gr.Textbox(label="Label for new mask")
                create_new_button = gr.Button("Create New Mask", size="sm", variant="primary")
                
                gr.Markdown("---")
                gr.Markdown("### Display Options")
                with gr.Row():
                    show_labels_checkbox = gr.Checkbox(label="Show Labels", value=True)
                    show_bboxes_checkbox = gr.Checkbox(label="Show B-Boxes", value=True)
                bbox_thickness_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="B-Box Thickness")
                
                gr.Markdown("---")
                finish_button = gr.Button("Finish Session", variant="primary")
                
                # Hidden component to handle Enter key
                enter_key_trigger = gr.Button("", visible=False)

        def change_edit_mode(mode_value):
            """Show/hide appropriate interface based on edit mode."""
            if mode_value == "Points":
                return (
                    gr.update(visible=True),   # points_group
                    gr.update(visible=False),  # bbox_group
                    gr.update(visible=True),   # point_mode radio
                    gr.update(label="Click to add positive points (green) or negative points (red). Use Update Masks to apply changes.")  # image_display
                )
            elif mode_value == "Bounding Box":
                return (
                    gr.update(visible=False),  # points_group
                    gr.update(visible=True),   # bbox_group
                    gr.update(visible=False),  # point_mode radio
                    gr.update(label="Click two points to define opposite corners of a bounding box.")  # image_display
                )

        def delete_masks(current_detections, selected_indices_str):
            """Deletes selected masks from the state."""
            if not selected_indices_str:
                return current_detections, ui_render_display(current_detections), gr.update(choices=get_detection_choices(current_detections))

            # Get integer indices from the checkbox strings
            indices_to_delete = sorted([int(s.split(':')[0]) for s in selected_indices_str], reverse=True)
            
            for index in indices_to_delete:
                if 0 <= index < len(current_detections):
                    current_detections.pop(index)

            # Render only on display (downscaled) image
            new_image = ui_render_display(current_detections)
            new_choices = get_detection_choices(current_detections)
            return current_detections, new_image, gr.update(choices=new_choices, value=[])
        
        def handle_image_click(current_detections, point_mode_value, edit_mode_value, evt: gr.SelectData):
            """Unified handler for all image interactions based on edit mode."""
            try:
                # Debug the event data
                print(f"Event data: {evt}")
                print(f"Event index: {evt.index}")
                
                if evt.index is None:
                    gr.Warning("Could not get click coordinates.")
                    return current_detections, ui_render_display(current_detections), "No points added yet", "No bounding box drawn"
                
                # Handle different possible formats of evt.index
                if isinstance(evt.index, (list, tuple)) and len(evt.index) >= 2:
                    x, y = evt.index[0], evt.index[1]
                elif hasattr(evt, 'x') and hasattr(evt, 'y'):
                    x, y = evt.x, evt.y
                else:
                    gr.Warning(f"Unexpected event format: {evt.index}")
                    return current_detections, ui_render_display(current_detections), "No points added yet", "No bounding box drawn"
                
                if edit_mode_value == "Points":
                    # Handle point mode
                    is_positive = point_mode_value == "Positive (include)"
                    
                    point_type = "positive" if is_positive else "negative"
                    # Store only full-resolution coordinate
                    fx, fy = map_display_to_full(int(x), int(y))
                    modification_points[point_type].append((fx, fy))
                    
                    # Update points display
                    pos_points = modification_points["positive"]
                    neg_points = modification_points["negative"]
                    points_text = f"Positive: {len(pos_points)} points, Negative: {len(neg_points)} points"
                    
                    # Render image with points visible immediately but efficiently
                    rendered_image = render_image_with_points(
                        display_image_array,
                        current_detections,
                        modification_points,
                        map_full_to_display_fn=map_full_to_display,
                        map_box_fn=map_box_full_to_display
                    )
                    
                    gr.Info(f"Added {'positive' if is_positive else 'negative'} point at ({int(x)}, {int(y)})")
                    return current_detections, rendered_image, points_text, "No bounding box drawn"
                
                elif edit_mode_value == "Bounding Box":
                    # Handle bounding box mode
                    fx, fy = map_display_to_full(int(x), int(y))
                    bbox_points.append((fx, fy))
                    
                    # Keep only last 2 points
                    if len(bbox_points) > 2:
                        bbox_points.pop(0)
                    
                    if len(bbox_points) == 1:
                        bbox_text = f"First corner: ({bbox_points[0][0]}, {bbox_points[0][1]}). Click second corner."
                    elif len(bbox_points) == 2:
                        bbox_text = f"Box: ({bbox_points[0][0]}, {bbox_points[0][1]}) to ({bbox_points[1][0]}, {bbox_points[1][1]})"
                    else:
                        bbox_text = "No bounding box drawn"
                    preview_image = render_image_with_bbox_preview(
                        display_image_array,
                        current_detections,
                        bbox_points,
                        map_full_to_display_fn=map_full_to_display,
                        map_box_fn=map_box_full_to_display
                    )
                    return current_detections, preview_image, "No points added yet", bbox_text
                return current_detections, ui_render_display(current_detections), "No points added yet", "No bounding box drawn"
                
                #else:
                #    # Other modes (Sketch) don't use click handling
                #    return current_detections, render_image_with_masks(display_image_array, current_detections), "No points added yet", "No bounding box drawn"
                
            except Exception as e:
                gr.Warning(f"Error handling image click: {str(e)}")
                print(f"Error in handle_image_click: {e}")
                import traceback
                traceback.print_exc()
                return current_detections, ui_render_display(current_detections), "Error handling click", "Error"
        
        def clear_points(current_detections):
            """Clear all modification points."""
            modification_points["positive"].clear()
            modification_points["negative"].clear()
            return current_detections, ui_render_display(current_detections), "No points added yet", "No bounding box drawn"

        def undo_last_point(current_detections):
            """Remove the last added point (positive or negative)."""
            # Find which list has the most recent point by checking if any exist
            pos_points = modification_points["positive"]
            neg_points = modification_points["negative"]
            
            # Simple approach: remove from negative first if it exists, otherwise from positive
            if neg_points:
                neg_points.pop()
            elif pos_points:
                pos_points.pop()
            
            # Update display
            points_text = f"Positive: {len(pos_points)} points, Negative: {len(neg_points)} points"
            if not pos_points and not neg_points:
                points_text = "No points added yet"
            
            # Re-render with remaining points
            rendered_image = render_image_with_points(
                display_image_array,
                current_detections,
                modification_points,
                map_full_to_display_fn=map_full_to_display,
                map_box_fn=map_box_full_to_display
            )
            
            return current_detections, rendered_image, points_text, "No bounding box drawn"

        def resolve_mask_conflicts(updated_detections, updated_indices):
            """
            Resolve overlapping masks by giving priority to recently updated masks.
            Removes overlapping pixels from non-updated masks.
            """
            try:
                for updated_idx in updated_indices:
                    if updated_idx >= len(updated_detections):
                        continue
                        
                    updated_detection = updated_detections[updated_idx]
                    if updated_detection is None or updated_detection.mask is None:
                        continue
                        
                    updated_mask = updated_detection.mask
                    
                    # Check for conflicts with other masks
                    indices_to_remove = []
                    for other_idx, other_detection in enumerate(updated_detections):
                        if (other_idx == updated_idx or 
                            other_detection is None or 
                            other_detection.mask is None):
                            continue
                        
                        try:
                            # Ensure masks have the same shape for comparison
                            if updated_mask.shape != other_detection.mask.shape:
                                print(f"Warning: Mask shape mismatch - updated: {updated_mask.shape}, other: {other_detection.mask.shape}")
                                continue
                            
                            # Find overlapping pixels
                            overlap = np.logical_and(updated_mask, other_detection.mask)
                            
                            if np.any(overlap):
                                # Remove overlapping pixels from the other mask
                                new_other_mask = np.logical_and(other_detection.mask, ~overlap)
                                
                                # Update the other detection if it still has pixels
                                if np.any(new_other_mask):
                                    # Recalculate bounding box for the reduced mask
                                    y_coords, x_coords = np.where(new_other_mask)
                                    new_box = BoundingBox(
                                        xmin=int(np.min(x_coords)),
                                        ymin=int(np.min(y_coords)),
                                        xmax=int(np.max(x_coords)),
                                        ymax=int(np.max(y_coords))
                                    )
                                    
                                    updated_detections[other_idx] = DetectionResult(
                                        score=other_detection.score,
                                        label=other_detection.label,
                                        box=new_box,
                                        mask=new_other_mask,
                                        logits=other_detection.logits
                                    )
                                    print(f"Resolved conflict: Mask {other_idx} ({other_detection.label}) reduced due to overlap with mask {updated_idx}")
                                else:
                                    # If the other mask is completely consumed, mark for removal
                                    print(f"Resolved conflict: Mask {other_idx} ({other_detection.label}) completely consumed by mask {updated_idx}")
                                    indices_to_remove.append(other_idx)
                        except Exception as e:
                            print(f"Error processing conflict between masks {updated_idx} and {other_idx}: {e}")
                            continue
                    
                    # Remove completely consumed masks (in reverse order to maintain indices)
                    for idx in sorted(indices_to_remove, reverse=True):
                        if idx < len(updated_detections):
                            updated_detections.pop(idx)
                        
            except Exception as e:
                print(f"Error in resolve_mask_conflicts: {e}")
                import traceback
                traceback.print_exc()
            
        def update_masks(current_detections, selected_indices_str):
            """Updates masks using points and segmentation model."""
            if not modification_points["positive"] and not modification_points["negative"]:
                gr.Warning("Please add some points first by clicking on the image.")
                return current_detections, ui_render_display(current_detections), "No points added yet", gr.update(choices=get_detection_choices(current_detections)), "No bounding box drawn"
            
            if not selected_indices_str:
                gr.Warning("Please select masks to modify first.")
                return current_detections, ui_render_display(current_detections), "No points added yet", gr.update(choices=get_detection_choices(current_detections)), "No bounding box drawn"
            
            try:
                # Get indices to modify
                indices_to_modify = [int(s.split(':')[0]) for s in selected_indices_str]
                
                # Run segmentation with points for each selected mask
                updated_count = 0
                original_indices = indices_to_modify.copy()  # Keep track for conflict resolution
                successfully_updated_indices = []  # Track which indices were actually updated
                
                for index in indices_to_modify:
                    if 0 <= index < len(current_detections):
                        detection = current_detections[index]
                        if detection is None:
                            continue
                        # Determine if expansion points lie outside current bbox
                        pxs = [p[0] for p in modification_points["positive"]]
                        pys = [p[1] for p in modification_points["positive"]]
                        expanded_box = detection.box
                        use_logits = detection.logits is not None
                        if modification_points["positive"]:
                            outside = False
                            for px, py in modification_points["positive"]:
                                if not (detection.box.xmin <= px <= detection.box.xmax and detection.box.ymin <= py <= detection.box.ymax):
                                    outside = True
                                    break
                            if outside:
                                # Expand box to include all positive points plus margin
                                margin = 20
                                new_xmin = int(max(0, min(detection.box.xmin, min(pxs) - margin))) if pxs else detection.box.xmin
                                new_ymin = int(max(0, min(detection.box.ymin, min(pys) - margin))) if pys else detection.box.ymin
                                new_xmax = int(min(image_array.shape[1]-1, max(detection.box.xmax, max(pxs) + margin))) if pxs else detection.box.xmax
                                new_ymax = int(min(image_array.shape[0]-1, max(detection.box.ymax, max(pys) + margin))) if pys else detection.box.ymax
                                expanded_box = BoundingBox(xmin=new_xmin, ymin=new_ymin, xmax=new_xmax, ymax=new_ymax)
                                use_logits = False  # drop logits to let SAM search larger region

                        temp_detection_dict = {
                            "score": detection.score,
                            "label": detection.label,
                            "box": {
                                "xmin": expanded_box.xmin,
                                "ymin": expanded_box.ymin,
                                "xmax": expanded_box.xmax,
                                "ymax": expanded_box.ymax
                            },
                            # Only pass logits if not expanding beyond original box
                            "logits": detection.logits if use_logits else None,
                            "mask": detection.mask,
                        }

                        new_masks = segment_with_points(
                            image_array,
                            [temp_detection_dict],
                            modification_points,
                            config,
                            segment_function
                        )
                        
                        if new_masks and len(new_masks) > 0:
                            # Update the detection with new mask
                            current_detections[index] = new_masks[0]
                            successfully_updated_indices.append(index)
                            updated_count += 1
                
                # Resolve conflicts between updated masks and existing masks
                # Only resolve conflicts if we added positive points (expansion case)
                if updated_count > 0:
                    if modification_points["positive"]:
                        # Only resolve conflicts when expanding masks (positive points)
                        # Use successfully updated indices to avoid index errors
                        resolve_mask_conflicts(current_detections, successfully_updated_indices)
                        gr.Info(f"Updated {updated_count} mask(s) using {len(modification_points['positive'])} positive and {len(modification_points['negative'])} negative points. Resolved overlaps.")
                    else:
                        # Just negative points - no conflict resolution needed
                        gr.Info(f"Updated {updated_count} mask(s) using {len(modification_points['negative'])} negative points.")
                else:
                    gr.Warning("No masks were updated. Please try different points.")
                
            except Exception as e:
                gr.Warning(f"Error updating masks: {str(e)}")
                print(f"Full error details: {e}")
                print(f"Points being used: positive={modification_points['positive']}, negative={modification_points['negative']}")
                print(f"Selected indices: {selected_indices_str}")
                import traceback
                traceback.print_exc()
            
            # Clear points only if we actually updated something
            if updated_count > 0:
                modification_points["positive"].clear()
                modification_points["negative"].clear()
            
            new_image = ui_render_display(current_detections)
            # Update checkbox choices since masks might have been removed during conflict resolution
            new_choices = get_detection_choices(current_detections)
            points_msg = "No points added yet" if updated_count > 0 else f"Retained points: +{len(modification_points['positive'])}/-{len(modification_points['negative'])}" 
            return current_detections, new_image, points_msg, gr.update(choices=new_choices, value=[]), "No bounding box drawn"
        
        def create_new_mask(current_detections, new_label):
            """Creates a new mask using points and segmentation model."""
            if not new_label or not new_label.strip():
                gr.Warning("Please enter a label for the new mask.")
                return current_detections, ui_render_display(current_detections), "No points added yet", "", gr.update(), "No bounding box drawn"
            
            if not modification_points["positive"]:
                gr.Warning("Please add at least one positive point by clicking on the image.")
                return current_detections, ui_render_display(current_detections), "No points added yet", new_label, gr.update(), "No bounding box drawn"
            
            try:
                # Create a bounding box around all points for initial detection
                all_points = modification_points["positive"] + modification_points["negative"]
                if not all_points:
                    gr.Warning("No points available for creating new mask.")
                    return current_detections, ui_render_display(current_detections), "No points added yet", new_label, gr.update(), "No bounding box drawn"
                
                # Calculate bounding box from points
                x_coords = [p[0] for p in all_points]
                y_coords = [p[1] for p in all_points]
                
                margin = 20  # Add some margin around the points
                xmin = max(0, min(x_coords) - margin)
                ymin = max(0, min(y_coords) - margin)
                xmax = min(image_array.shape[1], max(x_coords) + margin)
                ymax = min(image_array.shape[0], max(y_coords) + margin)
                
                # Create temporary detection for segmentation
                # Sanitize label: replace spaces and periods with underscores
                sanitized_label = new_label.strip().replace(" ", "_").replace(".", "_")
                temp_detection_dict = {
                    "score": 1.0,
                    "label": sanitized_label,
                    "box": {
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax
                    }
                }
                
                # Run segmentation with points
                new_masks = segment_with_points(
                    image_array,
                    [temp_detection_dict],
                    modification_points,
                    config,
                    segment_function
                )
                
                if new_masks and len(new_masks) > 0:
                    # Add new detection to the list
                    current_detections.append(new_masks[0])
                    
                    # Resolve conflicts with existing masks (new mask gets priority)
                    new_mask_index = len(current_detections) - 1
                    resolve_mask_conflicts(current_detections, [new_mask_index])
                    
                    gr.Info(f"Created new mask '{new_label}' using {len(modification_points['positive'])} positive and {len(modification_points['negative'])} negative points. Resolved overlaps.")
                else:
                    gr.Warning("Failed to create new mask. Please try different points.")
                
            except Exception as e:
                gr.Warning(f"Error creating new mask: {str(e)}")
                return current_detections, ui_render_display(current_detections), "No points added yet", new_label, gr.update(), "No bounding box drawn"
            
            # Clear points and label after creation
            modification_points["positive"].clear()
            modification_points["negative"].clear()
            
            new_image = ui_render_display(current_detections)
            return current_detections, new_image, "No points added yet", "", gr.update(choices=get_detection_choices(current_detections)), "No bounding box drawn"

        def finish_session(current_detections):
            """Saves the final state and signals completion."""
            global final_detections_state
            final_detections_state = current_detections
            session_finished["finished"] = True
            gr.Info("Session finished! You can close this browser tab now. The script will continue automatically.")
            # No return value since no outputs are specified
        
        def clear_bbox(current_detections):
            """Clear bounding box editor."""
            bbox_points.clear()
            return current_detections, ui_render_display(current_detections), "No bounding box drawn"
        
        def handle_enter_key(current_detections, edit_mode_value, selected_indices_str, new_label):
            """Handle Enter key press - trigger appropriate action based on current mode."""
            try:
                if edit_mode_value == "Bounding Box":
                    # Trigger bounding box application
                    return apply_bbox(current_detections, new_label, selected_indices_str)
                elif edit_mode_value == "Points":
                    # Trigger points update
                    if modification_points["positive"] or modification_points["negative"]:
                        return update_masks(current_detections, selected_indices_str)
                    else:
                        # Try to create new mask if label provided
                        if new_label and new_label.strip():
                            return create_new_mask(current_detections, new_label)
                        else:
                            gr.Info("Add some points first or enter a label for new mask")
                            return current_detections, ui_render_display(current_detections), "No points added yet", new_label, gr.update(), "No bounding box drawn"
                else:
                    # Sketch mode or other - no direct action
                    gr.Info("Enter key not supported in this mode")
                    return current_detections, ui_render_display(current_detections), "No points added yet", new_label, gr.update(), "No bounding box drawn"
            except Exception as e:
                gr.Warning(f"Error handling Enter key: {str(e)}")
                return current_detections, ui_render_display(current_detections), "Error", new_label, gr.update(), "Error"
            
        def apply_bbox(current_detections, new_label, selected_indices_str):
            """Create new mask or modify existing masks from bounding box using SAM."""
            if len(bbox_points) != 2:
                gr.Warning("Please click exactly 2 points to define the bounding box.")
                return current_detections, ui_render_display(current_detections), new_label, f"Need 2 points, have {len(bbox_points)}", gr.update()
            
            # Check if we're modifying existing masks or creating new one
            is_modifying_existing = selected_indices_str and len(selected_indices_str) > 0
            
            if not is_modifying_existing and (not new_label or not new_label.strip()):
                gr.Warning("Please enter a label for new mask or select existing masks to modify.")
                return current_detections, ui_render_display(current_detections), new_label, "Please enter label or select masks", gr.update()
            
            try:
                # Create bounding box from two points
                x1, y1 = bbox_points[0]
                x2, y2 = bbox_points[1]
                
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)
                
                # Load SAM model
                global segmentator_predictor
                load_segmentation_model(config)
                
                if is_modifying_existing:
                    # Modify existing masks using bounding box
                    indices_to_modify = [int(s.split(':')[0]) for s in selected_indices_str]
                    updated_count = 0
                    successfully_updated_indices = []
                    
                    for index in indices_to_modify:
                        if 0 <= index < len(current_detections):
                            detection = current_detections[index]
                            if detection is None:
                                continue
                            
                            # Use SAM with bounding box to refine the existing mask
                            if segmentator_predictor is not None:
                                # Predictor image assumed already set once.
                                
                                input_box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                                
                                # Use existing logits if available for refinement
                                mask_input = None
                                if detection.logits is not None:
                                    existing_logits = detection.logits
                                    if isinstance(existing_logits, torch.Tensor):
                                        existing_logits = existing_logits.detach().cpu().numpy()
                                    existing_logits = np.array(existing_logits)
                                    if existing_logits.ndim == 2:
                                        existing_logits = existing_logits[None, :, :]
                                    elif existing_logits.ndim == 3 and existing_logits.shape[0] != 1:
                                        existing_logits = existing_logits[:1, :, :]
                                    mask_input = existing_logits.astype(np.float32)
                                
                                masks, scores, logits = segmentator_predictor.predict(
                                    point_coords=None,
                                    point_labels=None,
                                    box=input_box,
                                    mask_input=mask_input,
                                    multimask_output=False,
                                )
                                
                                if len(masks) > 0:
                                    new_mask = masks[0]
                                    
                                    # Union with existing mask for additive behavior
                                    if detection.mask is not None:
                                        if new_mask.shape == detection.mask.shape:
                                            new_mask = np.logical_or(new_mask, detection.mask)
                                    
                                    # Create new bounding box
                                    if np.any(new_mask):
                                        y_coords, x_coords = np.where(new_mask)
                                        new_box = BoundingBox(
                                            xmin=int(np.min(x_coords)),
                                            ymin=int(np.min(y_coords)),
                                            xmax=int(np.max(x_coords)),
                                            ymax=int(np.max(y_coords))
                                        )
                                        
                                        current_detections[index] = DetectionResult(
                                            score=scores[0],
                                            label=detection.label,
                                            box=new_box,
                                            mask=new_mask,
                                            logits=logits[0]
                                        )
                                        successfully_updated_indices.append(index)
                                        updated_count += 1
                    
                    if updated_count > 0:
                        # Resolve conflicts for modified masks
                        resolve_mask_conflicts(current_detections, successfully_updated_indices)
                        bbox_points.clear()
                        gr.Info(f"Updated {updated_count} mask(s) using bounding box. Resolved overlaps.")
                        new_image = ui_render_display(current_detections)
                        # Keep the same selections after updating
                        return current_detections, new_image, new_label, "No bounding box drawn", gr.update(choices=get_detection_choices(current_detections), value=selected_indices_str)
                    else:
                        gr.Warning("No masks were updated.")
                        return current_detections, ui_render_display(current_detections), new_label, "No bounding box drawn", gr.update()
                
                else:
                    # Create new mask using bounding box
                    if segmentator_predictor is not None:
                        # Predictor image assumed already set once.
                        
                        input_box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                        
                        masks, scores, logits = segmentator_predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_box,
                            multimask_output=False,
                        )
                        
                        if len(masks) > 0:
                            sam_mask = masks[0]
                            
                            if np.any(sam_mask):
                                y_coords, x_coords = np.where(sam_mask)
                                new_box = BoundingBox(
                                    xmin=int(np.min(x_coords)),
                                    ymin=int(np.min(y_coords)),
                                    xmax=int(np.max(x_coords)),
                                    ymax=int(np.max(y_coords))
                                )
                                
                                new_detection = DetectionResult(
                                    score=scores[0],
                                    label=new_label.strip(),
                                    box=new_box,
                                    mask=sam_mask,
                                    logits=logits[0]
                                )
                            else:
                                # Fallback to rectangular mask
                                mask = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=bool)
                                mask[ymin:ymax, xmin:xmax] = True
                                new_box = BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
                                new_detection = DetectionResult(
                                    score=1.0,
                                    label=new_label.strip(),
                                    box=new_box,
                                    mask=mask
                                )
                        else:
                            # Fallback to rectangular mask
                            mask = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=bool)
                            mask[ymin:ymax, xmin:xmax] = True
                            new_box = BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
                            new_detection = DetectionResult(
                                score=1.0,
                                label=new_label.strip(),
                                box=new_box,
                                mask=mask
                            )
                    else:
                        # Fallback if SAM not available
                        mask = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=bool)
                        mask[ymin:ymax, xmin:xmax] = True
                        new_box = BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
                        new_detection = DetectionResult(
                            score=1.0,
                            label=new_label.strip(),
                            box=new_box,
                            mask=mask
                        )
                    
                    current_detections.append(new_detection)
                    
                    # Resolve conflicts with existing masks
                    new_mask_index = len(current_detections) - 1
                    resolve_mask_conflicts(current_detections, [new_mask_index])
                    
                    bbox_points.clear()
                    gr.Info(f"Created new SAM mask '{new_label}' from bounding box. Resolved overlaps.")
                    
                    new_image = ui_render_display(current_detections)
                    return current_detections, new_image, "", "No bounding box drawn", gr.update(choices=get_detection_choices(current_detections), value=[])
                
            except Exception as e:
                gr.Warning(f"Error with bounding box operation: {str(e)}")
                print(f"Bounding box error: {e}")
                import traceback
                traceback.print_exc()
                return current_detections, ui_render_display(current_detections), new_label, f"Error: {str(e)}", gr.update()

        # Wire up the component events to the handler functions
        delete_button.click(
            fn=delete_masks,
            inputs=[detections_state, detection_checkboxes],
            outputs=[detections_state, image_display, detection_checkboxes]
        )
        
        # Handle image clicks for all modes
        image_display.select(
            fn=handle_image_click,
            inputs=[detections_state, point_mode, edit_mode],
            outputs=[detections_state, image_display, points_display, bbox_points_display]
        )
        
        clear_points_button.click(
            fn=clear_points,
            inputs=[detections_state],
            outputs=[detections_state, image_display, points_display, bbox_points_display]
        )
        
        undo_last_point_button.click(
            fn=undo_last_point,
            inputs=[detections_state],
            outputs=[detections_state, image_display, points_display, bbox_points_display]
        )
        
        update_masks_button.click(
            fn=update_masks,
            inputs=[detections_state, detection_checkboxes],
            outputs=[detections_state, image_display, points_display, detection_checkboxes, bbox_points_display]
        )
        
        create_new_button.click(
            fn=create_new_mask,
            inputs=[detections_state, new_label_textbox],
            outputs=[detections_state, image_display, points_display, new_label_textbox, detection_checkboxes, bbox_points_display]
        )
        
        finish_button.click(
            fn=finish_session,
            inputs=[detections_state]
        )
        
        # Edit mode change handler
        edit_mode.change(
            fn=change_edit_mode,
            inputs=[edit_mode],
            outputs=[points_group, bbox_group, point_mode, image_display]
        )
        
        # Bounding box mode handlers (will be handled by main image)
        clear_bbox_button.click(
            fn=clear_bbox,
            inputs=[detections_state],
            outputs=[detections_state, image_display, bbox_points_display]
        )
        
        apply_bbox_button.click(
            fn=apply_bbox,
            inputs=[detections_state, new_label_textbox, detection_checkboxes],
            outputs=[detections_state, image_display, new_label_textbox, bbox_points_display, detection_checkboxes]
        )
        
        # Handle Enter key in the label textbox
        new_label_textbox.submit(
            fn=handle_enter_key,
            inputs=[detections_state, edit_mode, detection_checkboxes, new_label_textbox],
            outputs=[detections_state, image_display, points_display, new_label_textbox, detection_checkboxes, bbox_points_display]
        )

        # Event listeners for display options (live update)
        def update_display(dets, sl, sb, bt):
            # Update shared settings so all future renders use these values
            display_settings["show_labels"] = sl
            display_settings["show_bboxes"] = sb
            display_settings["bbox_thickness"] = bt
            return ui_render_display(dets, sl, sb, bt)
        
        for component in [show_labels_checkbox, show_bboxes_checkbox, bbox_thickness_slider]:
            component.change(
                fn=update_display,
                inputs=[detections_state, show_labels_checkbox, show_bboxes_checkbox, bbox_thickness_slider],
                outputs=image_display
            )

    # Launch the Gradio app in a separate thread to allow monitoring
    # When running remotely via SSH, VS Code can forward this port for you.

    
    import threading
    import time
    import random
    
    # Launch in non-blocking mode
    # random server port to avoid conflicts
    random_port = random.randint(7860, 8000)
    print(f"Gradio App is running at http://127.0.0.1:{random_port}")
    print("If using VS Code Remote, it should automatically forward this port.")

    demo.launch(
        server_name="127.0.0.1",
        server_port=random_port,
        share=False,
        show_error=True,
        quiet=False,
        prevent_thread_lock=True  # This allows the script to continue
    )
    
    # Monitor for session completion
    print("Waiting for session to complete...")
    while not session_finished["finished"]:
        time.sleep(1)  # Check every second
    
    # Close the demo and return results
    demo.close()
    print("Session completed, continuing script...")
    return final_detections_state


# Backward compatibility alias
def edit_segmentations(image_array: np.ndarray, initial_detections: list[DetectionResult]):
    """Backward compatibility wrapper for the old interface."""
    return edit_segmentations_interactive(image_array, initial_detections)