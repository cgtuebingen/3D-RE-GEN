# Base code
# Author of this notebook: [Eduardo Pacheco](https://huggingface.co/EduardoPacheco) - give him a follow on Hugging
#  Face!

# load modules
import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple
import yaml
import argparse

import cv2
import torch
import requests
import numpy as np
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline, AutoModel, SamHQModel, SamHQProcessor

from PIL import Image
import os

import shutil

from upscaler import Upscaler
from utils.global_utils import clear_output_directory, load_config, calculate_iou


from point_generators import get_random_point, get_entropy_points, get_distance_points, get_saliency_point

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    center: Optional[Tuple[int, int]] = None
    area: Optional[int] = None

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None
    
    def __post_init__(self):
        self.box.center = ((self.box.xmin + self.box.xmax) // 2, (self.box.ymin + self.box.ymax) // 2)
        self.box.area = (self.box.xmax - self.box.xmin) * (self.box.ymax - self.box.ymin)
    

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))
    





def expand_bbox(bbox, scale_factor=1.25, image=None):
    """
    Expand a bounding box by a scale factor.
    
    Args:
        bbox: List or tuple of [x_min, y_min, x_max, y_max]
        scale_factor: Factor by which to expand the box (1.0 = no change)
        image_width: Width of the image (to constrain box within image)
        image_height: Height (to constrain box within image)
    
    Returns:
        Expanded bounding box as [x_min, y_min, x_max, y_max]
    """
    x_min, y_min, x_max, y_max = bbox
    image_width, image_height = (image.size if image else (None, None))
    
    # Calculate width and height
    width = x_max - x_min
    height = y_max - y_min
    
    # Calculate center
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    # Calculate new width and height
    new_width = width * scale_factor
    new_height = height * scale_factor
    
    # Calculate new corners
    new_x_min = x_center - new_width / 2
    new_y_min = y_center - new_height / 2
    new_x_max = x_center + new_width / 2
    new_y_max = y_center + new_height / 2
    
    # Constrain within image boundaries if dimensions are provided
    if image_width and image_height:
        new_x_min = max(0, min(new_x_min, image_width))
        new_y_min = max(0, min(new_y_min, image_height))
        new_x_max = max(0, min(new_x_max, image_width))
        new_y_max = max(0, min(new_y_max, image_height))

    return [int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max)]



def filter_duplicate_detections(detections, iou_threshold=0.5):
    """
    Filter duplicate detections by keeping only the highest-scoring detection 
    when multiple detections have significant overlap.
    
    Args:
        detections: List of DetectionResult objects
        iou_threshold: Threshold for considering two boxes as duplicates
    
    Returns:
        List of filtered DetectionResult objects
    """
    if not detections:
        print("No detections to filter.")
        return []
    
    # Sort detections by score in descending order
    detections = sorted(detections, key=lambda x: x.score, reverse=True)
    filtered_detections = []
    
    while detections:
        # Take the highest scoring detection
        current = detections.pop(0)
        filtered_detections.append(current)
        
        # Remove detections with high IoU with the current detection
        detections = [d for d in detections
                     if calculate_iou(current.box.xyxy, d.box.xyxy) < iou_threshold]

    return filtered_detections



def create_points(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult], method: str = "random") -> List[List[int]]:
    """
    For each detection, extract the cropped region and generate a point using the specified method.
    Returns a list of points for each detection (format: [[(x1, y1)], [(x2, y2)], ...])

    method options:
    - "random": Random point within the mask
    - "max_entropy": Point with maximum entropy within the mask
    - "max_distance": Point with maximum distance from the mask boundary
    - "saliency": Point with maximum saliency within the cropped image region

    """
    points = []
    image_np = np.array(image) if isinstance(image, Image.Image) else image

    for detection in detection_results:
        box = detection.box
        mask = detection.mask
        label = detection.label
        # Crop the image region
        cropped_img = image_np[box.ymin:box.ymax, box.xmin:box.xmax]
        # Crop the mask region
        cropped_mask = mask[box.ymin:box.ymax, box.xmin:box.xmax] if mask is not None else None

        # Choose point generation method
        if method == "random":
            pt = get_random_point(cropped_mask)
        elif method == "max_entropy":
            # You need an input point for get_entropy_points; use center or random
            input_point = [cropped_mask.shape[1] // 2, cropped_mask.shape[0] // 2]
            pt = get_entropy_points(input_point, cropped_mask, cropped_img)
        elif method == "max_distance":
            input_point = [cropped_mask.shape[1] // 2, cropped_mask.shape[0] // 2]
            pt = get_distance_points(input_point, cropped_mask)
        elif method == "saliency":
            pt = get_saliency_point(cropped_img, cropped_mask)
        else:
            raise ValueError(f"Unknown point generation method: {method}, use one of: 'random', 'max_entropy', 'max_distance', 'saliency'.")

        # Convert point to absolute image coordinates
        #print(f"Generated point for {label}: {pt} in cropped region with box {box}")
        abs_pt = [pt[0] + box.xmin, pt[1] + box.ymin]
        #print(f"Absolute point for {label}: {abs_pt} in original image coordinates")

        points.append([abs_pt])  # SAM expects a list of points per detection

    return [points] # SAM pipeline expects a batch: [ [ [x1, y1] ], [ [x2, y2] ], ... ]

    

def annotate(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

def plot_detections(
    image: Union[Image.Image, np.ndarray],
    detections: List[DetectionResult],
    save_name: Optional[str] = None
) -> None:

    # increase the size of the figure
    plt.figure(figsize=(12, 8))
    annotated_image = annotate(image, detections)
    plt.imshow(annotated_image)
    plt.axis('off')
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()


def random_named_css_colors(num_colors: int) -> List[str]:
    """
    Returns a list of randomly selected named CSS colors.

    Args:
    - num_colors (int): Number of random colors to generate.

    Returns:
    - list: List of randomly selected named CSS colors.
    """
    # List of named CSS colors
    named_css_colors = [
        'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond',
        'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
        'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey',
        'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
        'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
        'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite',
        'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory',
        'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow',
        'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray',
        'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
        'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
        'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive',
        'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip',
        'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown',
        'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey',
        'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white',
        'whitesmoke', 'yellow', 'yellowgreen'
    ]

    # Sample random named CSS colors
    return random.sample(named_css_colors, min(num_colors, len(named_css_colors)))

def plot_detections_plotly(
    image: np.ndarray,
    detections: List[DetectionResult],
    class_colors: Optional[Dict[str, str]] = None
) -> None:
    # If class_colors is not provided, generate random colors for each class
    if class_colors is None:
        num_detections = len(detections)
        colors = random_named_css_colors(num_detections)
        class_colors = {}
        for i in range(num_detections):
            class_colors[i] = colors[i]


    fig = px.imshow(image)

    # Add bounding boxes
    shapes = []
    annotations = []
    for idx, detection in enumerate(detections):
        label = detection.label
        box = detection.box
        score = detection.score
        mask = detection.mask

        polygon = mask_to_polygon(mask)

        fig.add_trace(go.Scatter(
            x=[point[0] for point in polygon] + [polygon[0][0]],
            y=[point[1] for point in polygon] + [polygon[0][1]],
            mode='lines',
            line=dict(color=class_colors[idx], width=2),
            fill='toself',
            name=f"{label}: {score:.2f}"
        ))

        xmin, ymin, xmax, ymax = box.xyxy
        shape = [
            dict(
                type="rect",
                xref="x", yref="y",
                x0=xmin, y0=ymin,
                x1=xmax, y1=ymax,
                line=dict(color=class_colors[idx])
            )
        ]
        annotation = [
            dict(
                x=(xmin+xmax) // 2, y=(ymin+ymax) // 2,
                xref="x", yref="y",
                text=f"{label}: {score:.2f}",
            )
        ]

        shapes.append(shape)
        annotations.append(annotation)

    # Update layout
    button_shapes = [dict(label="None",method="relayout",args=["shapes", []])]
    button_shapes = button_shapes + [
        dict(label=f"Detection {idx+1}",method="relayout",args=["shapes", shape]) for idx, shape in enumerate(shapes)
    ]
    button_shapes = button_shapes + [dict(label="All", method="relayout", args=["shapes", sum(shapes, [])])]

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        # margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        updatemenus=[
            dict(
                type="buttons",
                direction="up",
                buttons=button_shapes
            )
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Show plot
    fig.show()



def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")

    return image

def get_boxes(results: DetectionResult) -> List[List[float]]:
    boxes = []
    for result in results:
        boxes.append(result.box.xyxy)

    return [boxes]

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

##################################################################
# Grounded SAM
##################################################################

def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
    detector_id: Optional[str] = None,
    device="cpu"
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    device = torch.device(device) if torch.cuda.is_available() else "cpu"
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"

    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

    labels = [label if label.endswith(".") else label+"." for label in labels]

    results = object_detector(image,  candidate_labels=labels, threshold=threshold)
    
    results = [DetectionResult.from_dict(result) for result in results]
    # expand results bounding box by 3 pixels
    

    return results

def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
    segmenter_id: Optional[str] = None,
    device="cpu",
    use_points: bool = False,
    config=None,
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    device = torch.device(device) if torch.cuda.is_available() and device!="cpu" else "cpu"
    point_method = config.get("point_method", "random")

    # ipdated to https://huggingface.co/docs/transformers/main/model_doc/sam_hq
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)

    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results



def grounded_segmentation(
    image: Union[Image.Image, str],
    labels: List[str],
    threshold: float = 0.3,
    polygon_refinement: bool = False,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None,
    device: Optional[str] = None,
    config= None,
) -> Tuple[np.ndarray, List[DetectionResult]]:
    if isinstance(image, str):
        image = load_image(image)

    use_points = config.get("use_points", False)

    # Run grounded DINO to detect objects in the image
    detections = detect(image, labels, threshold, detector_id, device=device)
    print("Amount Unfiltered Detections and score:", len(detections), [(d.label, d.score) for d in detections])

    # Filter Detections if bbox are overlaying
    detections = filter_duplicate_detections(detections, iou_threshold=config.get("iou_threshold", 0.5))
    print("Amount Filtered Detections and score:", len(detections), [(d.label, d.score) for d in detections])

    # if use points do double run else single run, double run without points first to create masks needed for second run point generation
    if use_points:
        print("Using points for segmentation... Starting double run.")
        detections = segment(image, detections, polygon_refinement, segmenter_id, device=device, config=config, use_points=False)
        print("First run completed. Now refining with points...")
        detections = segment(image, detections, polygon_refinement, segmenter_id, device=device, config=config, use_points=True)
    else:
        print("Using bounding boxes only for segmentation... Starting single run.")
        detections = segment(image, detections, polygon_refinement, segmenter_id, device=device, config=config, use_points=False)

    return np.array(image), detections


##################################################################
def convert_and_clean_webp(webp_path, output_path):
    # Load image with alpha if present
    img = cv2.imread(webp_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Failed to load image from {webp_path}")

    # Handle RGBA
    if img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        alpha = a.astype(float) / 255.0
        # Blend with white background
        white = np.ones_like(img[:, :, :3], dtype=float) * 255
        rgb = cv2.merge((r, g, b)).astype(float)
        blended = rgb * alpha[..., None] + white * (1 - alpha[..., None])
        img_rgb = blended.astype(np.uint8)
    else:
        # No alpha: convert BGR to RGB
        b, g, r = cv2.split(img)
        img_rgb = cv2.merge((r, g, b))

        # Replace pure black background with white
        mask_black = np.all(img_rgb == [0, 0, 0], axis=-1)
        img_rgb[mask_black] = [255, 255, 255]

    # increase brightness by 20%
    #img_rgb = cv2.convertScaleAbs(img_rgb, alpha=1.2, beta=0)
    # Save as PNG using Pillow for consistency
    Image.fromarray(img_rgb).save(output_path, format="PNG")


########################
# Save segmented objects


def save_masked_findings(image_array: np.ndarray, detections: List[DetectionResult], output_dir: str = "masked_findings", **kwargs) -> None:
    """
    Save individual findings (masked regions) from the image based on segmentation masks.

    Args:
        image_array (np.ndarray): The original image as a NumPy array.
        detections (List[DetectionResult]): List of detection results with masks.
        output_dir (str): Directory to save the masked findings.
        **kwargs: Additional arguments for saving options (e.g., padding).

    """
    config = kwargs.get("config", {})
    padding = config.get("findings_padding", 0)  # Default padding if not specified

    # Remove the output directory if it exists, then recreate it
    os.makedirs(output_dir, exist_ok=True)

    # Remove all contents of the output directory
    clear_output_directory(output_dir)

    # create subdirectory for fullSize images and cropped images, remove prexisting directories
    full_size_dir = os.path.join(output_dir, "fullSize")
    os.makedirs(full_size_dir, exist_ok=True)

    cropped_dir = os.path.join(output_dir, "cropped")
    os.makedirs(cropped_dir, exist_ok=True)


    for idx, detection in enumerate(detections):
        mask = detection.mask
        box = detection.box
        label = detection.label.replace(" ", "_").replace(".", "_")
        # get bbox centre
        centre = detection.box.center
        #print(f"Detection {idx + 1}: {label} at {centre} with area {detection.box.area}")

        print(f"Processing {label} with score {detection.score:.2f}")
        
        if mask is not None:
            # Apply the mask to the image
            masked_image = np.zeros_like(image_array, dtype=np.uint8)
            masked_image[mask > 0] = image_array[mask > 0]
            # set the rest of the image to white
            masked_image[mask == 0] = [255, 255, 255]

            # # Add an alpha channel
            # alpha_channel = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
            # alpha_channel[mask > 0] = 255  # Set alpha to 255 for the masked region

            # # Combine RGB and alpha channels
            # masked_image_with_alpha = np.dstack((masked_image, alpha_channel))

            # Save the masked image with transparency
            masked_out = Image.fromarray(masked_image)# , mode="RGBA")
            masked_out.save(os.path.join(full_size_dir, f"{label}_{centre}.png"))

            # Crop the masked image to the bounding box with padding
            xmin = max(box.xmin - padding, 0)  # Add padding, ensure within bounds
            ymin = max(box.ymin - padding, 0)
            xmax = min(box.xmax + padding, image_array.shape[1])
            ymax = min(box.ymax + padding, image_array.shape[0])

            cropped_image = masked_image[ymin:ymax, xmin:xmax]

            # Convert to PIL Image and save
            cropped_pil = Image.fromarray(cropped_image)#, mode="RGBA")
            cropped_pil.save(os.path.join(cropped_dir, f"{label}_{centre}.png"))



def main():
    # Load configuration
    parser = argparse.ArgumentParser(description="Run segmentation script with config file.")
    parser.add_argument("--config", default="../src/config.yaml", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)

    image_url = config["input_image"]
    labels = config["labels"]
    threshold = config["threshold"]
    detector_id = config["detector_id"]
    segmenter_id = config["segmenter_id"]
    device = config["device"]
    output_dir = config["output_seg"]


    # Print segmenetation with detector and segmenter
    print(f"Running segmentation with detector: {detector_id}, segmenter: {segmenter_id}, device: {config["device"]}")

    upscaler = Upscaler(model_name="SD", device=device)

    if image_url.endswith(".webp"):
        cleaned_png_path = os.path.join(config["temp"], "converted_from_webp.png")
        convert_and_clean_webp(image_url, cleaned_png_path)
        image_url = cleaned_png_path

        # save over config.yaml image_url with cleaned_png_path
        config["image_url"] = cleaned_png_path
        
        # with open(args.config, "w") as config_file:
        #     yaml.dump(config, config_file)


    # if upscale_input_image is True, upscale the input image
    if config.get("upscale_input_image", False):
        print("Upscaling input image...")
        input_image = load_image(image_url)
        upscaled_image = upscaler(
            input_image, 
            size=config["size"], 
            num_inference_steps=config["num_inference_steps"], 
            guidance_scale=config["guidance_scale"]
        )
        # Save the upscaled image
        upscaled_image.save(os.path.join(config["temp"], "temp_upscaled_image.png"))
        image_url = os.path.join(config["temp"], "temp_upscaled_image.png")

        # save to image_url config
        config["image_url"] = image_url


    # Run segmentation
    image_array, detections = grounded_segmentation(
        image=image_url,
        labels=labels,
        threshold=threshold,
        polygon_refinement=False,
        detector_id=detector_id,
        segmenter_id=segmenter_id,
        device=device,
        config=config,
    )
    plot_detections(image_array, detections, "../output/box_segmented_image.png")
    # plot_detections_plotly(image_array, detections)
    save_masked_findings(
        image_array=image_array, 
        detections=detections, 
        output_dir=output_dir, 
        config=config
        )


    print("Finished processing. Upscaling images...",
        "##################################################################################################################")

    # iterate over all images in segmentor/findings/cropped/
    path = os.path.join(output_dir, 'cropped')
    # if path does not exist, throw error
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} does not exist.")

    # outpath
    outpath = os.path.join(output_dir, 'upscaled', 'cropped')
    # if outpath does not exist, create it
    os.makedirs(outpath, exist_ok=True)

    for filename in os.listdir(path):
        if filename.endswith('.png'):
            control_image = load_image(os.path.join(path, filename))
            upscaled_image = upscaler(
                control_image, 
                size=config["size"], 
                num_inference_steps=config["num_inference_steps"], 
                guidance_scale=config["guidance_scale"]
                )

            # save the upscaled image
            upscaled_image.save(os.path.join(outpath, filename))


if __name__ == "__main__":
    main()