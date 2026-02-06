from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig
from PIL import Image, ImageOps
from io import BytesIO
import os
import argparse
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import cv2
from utils.global_utils import load_config, clear_output_directory, extract_AQ_object
from rembg import remove, new_session


def make_bg_removal_less_aggressive(
    img_rgba, conservative_threshold=80, fill_holes_size=5, smooth_edges=True
):
    """
    Make background removal less aggressive by:
    1. Lowering the alpha threshold (keep more pixels)
    2. Filling small holes in the object
    3. Expanding the mask slightly to recover lost object parts

    Args:
        img_rgba: PIL Image in RGBA format
        conservative_threshold: Lower threshold keeps more object pixels (0-255)
        fill_holes_size: Size of holes to fill in the object
        smooth_edges: Whether to smooth edges while preserving detail

    Returns:
        PIL Image with less aggressive background removal
    """
    img_array = np.array(img_rgba)
    if img_array.shape[2] != 4:
        return img_rgba

    # Extract alpha channel
    alpha = img_array[:, :, 3]

    # Step 1: Use a much more conservative threshold to keep more object pixels
    # Instead of binary cutoff, use gradual preservation
    alpha_conservative = np.where(
        alpha > conservative_threshold,
        255,
        np.where(alpha > conservative_threshold // 2, alpha * 2, 0),
    ).astype(np.uint8)

    # Step 2: Fill holes that might be incorrectly removed object parts
    if fill_holes_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (fill_holes_size, fill_holes_size)
        )
        # Close operation fills small holes
        alpha_conservative = cv2.morphologyEx(
            alpha_conservative, cv2.MORPH_CLOSE, kernel
        )

        # Dilate slightly to recover potentially lost object edges
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha_conservative = cv2.dilate(alpha_conservative, small_kernel, iterations=1)

    # Step 3: Smooth edges if requested, but preserve object detail
    if smooth_edges:
        # Use bilateral filter to smooth while preserving edges
        alpha_conservative = cv2.bilateralFilter(alpha_conservative, 5, 50, 50)

    # Replace alpha channel
    img_array[:, :, 3] = alpha_conservative

    return Image.fromarray(img_array)


def sharpen_alpha_edges(
    img_rgba, alpha_threshold=100, blur_kernel=3, sharpen_strength=2.0
):
    """
    Apply edge sharpening to the alpha channel for crisper cutouts.
    Now with less aggressive processing to preserve object parts.

    Args:
        img_rgba: PIL Image in RGBA format
        alpha_threshold: Threshold for alpha binarization (0-255) - lowered from 128
        blur_kernel: Kernel size for morphological operations
        sharpen_strength: Strength of edge sharpening

    Returns:
        PIL Image with sharpened alpha channel
    """
    img_array = np.array(img_rgba)
    if img_array.shape[2] != 4:
        return img_rgba

    # Extract alpha channel
    alpha = img_array[:, :, 3]

    # Apply morphological operations to clean up the mask (less aggressive)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blur_kernel, blur_kernel))

    # Close small holes and smooth edges (gentler approach)
    alpha_clean = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)

    # Apply bilateral filter to smooth while preserving edges (less aggressive)
    alpha_smooth = cv2.bilateralFilter(
        alpha_clean, 7, 50, 50
    )  # Reduced filter strength

    # Create less aggressive binary mask - keep more gradual transitions
    alpha_sharp = np.where(
        alpha_smooth > alpha_threshold,
        255,
        np.where(alpha_smooth > alpha_threshold // 2, alpha_smooth, 0),
    ).astype(np.uint8)

    # Apply slight gaussian blur for anti-aliasing
    alpha_final = cv2.GaussianBlur(alpha_sharp, (3, 3), 0.5)

    # Replace alpha channel
    img_array[:, :, 3] = alpha_final

    return Image.fromarray(img_array)


def prepare_for_hunyuan(
    input_folder: str,
    output: str,
    size: int = 512,
    white_threshold: int = 245,
    margin_ratio: float = 0.08,
    upscale_factor: int = 4,
    conservative_mode: bool = True,
    use_AQ: bool = True,
) -> None:
    """
    Convert all images in `input_folder` (object on white background) to square 512x512 crops:
    - Upscale image for better ONNX processing
    - Remove background once to get clean object and bounds
    - Apply conservative post-processing to preserve object parts
    - Detect object by alpha channel for accurate centering
    - Compute tight bbox, expand to square with margin
    - Resize to `size` x `size` and save as PNG with sharp edges

    Args:
        conservative_mode: If True, apply less aggressive background removal
    """
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    os.makedirs(output, exist_ok=True)

    files = [
        f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in exts
    ]
    if not files:
        print(f"No images found in {input_folder}.")
        return

    # Initialize rembg session once for better performance
    print("Initializing background removal session...")
    try:
        # Force CPU providers for ONNX to avoid CUDA issues
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]  # Use CPU only to avoid cuDNN errors

        # Try different models for sharper edges - isnet-general-use often produces sharper results
        models_to_try = ["isnet-general-use", "u2netp", "u2net"]
        rembg_session = None

        for model in models_to_try:
            try:
                rembg_session = new_session(model, providers=providers)
                print(
                    f"Successfully initialized rembg with {model} model and CPU provider"
                )
                break
            except Exception as model_e:
                print(f"Failed to load {model}: {model_e}")
                continue

        if rembg_session is None:
            raise Exception("All models failed to load")

    except Exception as e:
        print(f"Warning: Failed to create optimized rembg session, using default: {e}")
        try:
            rembg_session = new_session("u2net")
            print("Fallback to default u2net session")
        except Exception as e2:
            print(f"Error: Could not initialize rembg session: {e2}")
            rembg_session = None

    processed = 0
    for fname in tqdm(files, desc="Processing images"):
        in_path = os.path.join(input_folder, fname)
        try:
            img = Image.open(in_path).convert("RGB")

            if use_AQ:
                # Extract only right side area where AQ object is located
                img = extract_AQ_object(img, target_width=img.width-img.height)


            # Step 1: Upscale image for better ONNX processing (sharper edges)
            original_size = img.size
            upscaled_size = (
                original_size[0] * upscale_factor,
                original_size[1] * upscale_factor,
            )

            # Use cv2 for high-quality upscaling (INTER_CUBIC for smooth upscaling)
            img_array = np.array(img)
            img_upscaled_array = cv2.resize(
                img_array, upscaled_size, interpolation=cv2.INTER_CUBIC
            )
            img_upscaled = Image.fromarray(img_upscaled_array)

            # Step 2: Remove background on upscaled image for better edge quality
            if rembg_session:
                img_removed = remove(
                    img_upscaled, session=rembg_session
                )  # RGBA with clean alpha
            else:
                img_removed = remove(img_upscaled)  # RGBA with clean alpha

            temp_arr = np.asarray(img_removed)

            # Use alpha channel to find actual object bounds
            if temp_arr.shape[2] == 4:
                alpha_channel = temp_arr[:, :, 3]
                non_transparent = alpha_channel > 10  # Low threshold for detection
            else:
                # Fallback to white detection if no alpha
                arr = np.asarray(img)
                non_transparent = (
                    ~(arr[:, :, 0] > white_threshold)
                    | ~(arr[:, :, 1] > white_threshold)
                    | ~(arr[:, :, 2] > white_threshold)
                )

            if non_transparent.sum() < 10:
                print(f"Skip (no object found): {fname}")
                continue

            # Find bounding box of actual object (from alpha)
            ys = np.where(non_transparent.any(axis=1))[0]
            xs = np.where(non_transparent.any(axis=0))[0]
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            x1, x2 = int(xs.min()), int(xs.max()) + 1

            # Square crop with margin (same logic as before)
            bw, bh = (x2 - x1), (y2 - y1)
            side = max(bw, bh)
            margin = int(max(2, round(side * margin_ratio)))
            side_sq = side + 2 * margin

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            x1s = int(np.floor(cx - side_sq / 2))
            y1s = int(np.floor(cy - side_sq / 2))
            x2s = x1s + side_sq
            y2s = y1s + side_sq

            # Pad the REMOVED image with transparent if crop goes out of bounds
            H, W = img_removed.height, img_removed.width
            pad_left = max(0, -x1s)
            pad_top = max(0, -y1s)
            pad_right = max(0, x2s - W)
            pad_bottom = max(0, y2s - H)

            if pad_left or pad_top or pad_right or pad_bottom:
                # Pad with transparent background for RGBA image
                img_removed = ImageOps.expand(
                    img_removed,
                    border=(pad_left, pad_top, pad_right, pad_bottom),
                    fill=(0, 0, 0, 0),
                )
                # shift crop coords into padded image
                x1s += pad_left
                x2s += pad_left
                y1s += pad_top
                y2s += pad_top

            # Final safe clamp
            x1s = max(0, x1s)
            y1s = max(0, y1s)
            x2s = min(img_removed.width, x2s)
            y2s = min(img_removed.height, y2s)

            # Step 3: Crop the already processed upscaled image (single rembg call)
            crop = img_removed.crop((x1s, y1s, x2s, y2s))

            # Step 4: Apply conservative post-processing if enabled
            if conservative_mode:
                # Make background removal less aggressive to preserve object parts
                crop_conservative = make_bg_removal_less_aggressive(
                    crop,
                    conservative_threshold=60,
                    fill_holes_size=20,
                    smooth_edges=True,
                )
                # Apply gentle edge sharpening
                crop_sharpened = sharpen_alpha_edges(
                    crop_conservative,
                    alpha_threshold=80,
                    blur_kernel=3,
                    sharpen_strength=1.5,
                )
            else:
                # Original aggressive processing
                crop_sharpened = sharpen_alpha_edges(
                    crop, alpha_threshold=100, blur_kernel=5, sharpen_strength=2.0
                )

            # Step 6: Resize to final size using high-quality downsampling for sharp edges
            # Convert to numpy for cv2 processing (preserving alpha channel)
            crop_array = np.array(crop_sharpened)
            if crop_array.shape[2] == 4:  # RGBA
                # Resize RGB and alpha separately for better quality
                rgb_part = crop_array[:, :, :3]
                alpha_part = crop_array[:, :, 3]

                rgb_resized = cv2.resize(
                    rgb_part, (size, size), interpolation=cv2.INTER_AREA
                )  # INTER_AREA is best for downscaling
                alpha_resized = cv2.resize(
                    alpha_part, (size, size), interpolation=cv2.INTER_AREA
                )

                # Combine back to RGBA
                final_array = np.dstack((rgb_resized, alpha_resized))
                resized = Image.fromarray(final_array)
            else:
                # Fallback to PIL if no alpha channel
                resized = crop_sharpened.resize((size, size), Image.LANCZOS)

            base, _ = os.path.splitext(fname)
            out_path = os.path.join(output, f"{base}.png")
            resized.save(out_path)
            processed += 1

        except Exception as e:
            print(f"Failed {fname}: {e}")

    print(f"Prepared {processed}/{len(files)} images into {output}")


# TODO : use google batch API
def process_image_worker(
    input_image_path: str,
    output_image_path: str, 
    base_prompt: str, 
    temperature: float = 1.0,
    model: str = "gemini-2.5-flash-image",
    config: dict = None
):
    """
    Worker function to process a single image using the GenAI client.
    This function is designed to be called by a multiprocessing pool.
    """
    filename = os.path.basename(input_image_path)

    try:
        # --- 1. Initialize client within the worker ---
        client = genai.Client()

        # --- 2. Prepare dynamic prompt from filename ---
        object_name = filename.split("__")[0].replace("_", " ")
        prompt_text = base_prompt.format(object=object_name)

        # --- 3. Load image and call API ---
        print(f"Processing: {filename} with prompt for '{object_name}'...")
        image = Image.open(input_image_path)

        response = client.models.generate_content(
            model= model,
            contents=[prompt_text, image],
            config=GenerateContentConfig(
                temperature=temperature, # 
                # candidate_count=1,
                # response_mime_type="application/json",
                top_p=config.get("genai_top_p", 0.9),
                # top_k=20,
                seed=config.get("seed", 12345),
                # image_config=types.ImageConfig(
                #     aspect_ratio ="1:1"
                # ),
                #max_output_tokens=500,
                # stop_sequences=["STOP!"],
                # presence_penalty=0.0,
                # frequency_penalty=0.0,
            )
        )  

        # --- 4. Save the output image ---
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                print(part.text)
            elif part.inline_data is not None:
                generated_image_data = BytesIO(part.inline_data.data)
                output_image = Image.open(generated_image_data)
                output_image.save(output_image_path)
                return f"Successfully processed and saved: {filename}"

        return f"Failed: No image data returned for {filename}"

    except Exception as e:
        return f"Error processing {filename}: {e}"


def main():
    """
    Main function to orchestrate the parallel image processing.
    """
    # Set environment variables to optimize ONNX Runtime performance and reduce errors
    os.environ["OMP_NUM_THREADS"] = "4"  # Limit OpenMP threads
    os.environ["ONNXRUNTIME_LOG_SEVERITY_LEVEL"] = (
        "3"  # Reduce log verbosity (ERROR level)
    )
    os.environ["ORT_DISABLE_TRT_FLASH_ATTENTION"] = (
        "1"  # Disable TensorRT flash attention
    )

    parser = argparse.ArgumentParser(
        description="Run batch inpainting with Google GenAI."
    )
    parser.add_argument(
        "--config",
        default="../src/config.yaml",
        type=str,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    config = load_config(args.config)



    # --- Prepare paths and files ---
    input_dir = os.path.join(config["output_seg_banana"], "outline")
    # if using bbox as input
    if config.get("use_bbox_as_input", False):
        input_dir = os.path.join(config["output_seg_banana"], "bbox")
    # if using AQ object extraction
    if config.get("use_AQ", True):
        input_dir = os.path.join(config["output_seg_banana"], "segmentation_layouts")

    output_dir = config["output_inp_banana"]

    os.makedirs(output_dir, exist_ok=True)

    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")

    # skip images containing wall, floor, ceiling
    images_to_process = [f for f in os.listdir(input_dir) if f.endswith(".png") and not any(x in f for x in ["wall", "floor", "ceiling", "room"])]

    if not images_to_process:
        print("No images found in the input directory. Exiting.")
        return

    # base_prompt = "Extract this marked {object}. Create a single render of the marked {object} with white background. \
    #    If there are objects like tables, decorations, chairs etc. in front, remove them. Fill any missing parts realistically. \
    #    No ground contact shadows, {object} just on white background."
    base_prompt = config.get(
        "banana_inpainting_prompt",
        "Extract this marked {object}. Create a single render of the marked {object} with white background."
    )

    if config.get("use_AQ", True):
        base_prompt = config["prompt_AQ"]

    print(f"Using base prompt: {base_prompt}")

    # --- Prepare tasks for the multiprocessing pool ---
    temperature = config.get("genai_temperature", 0.7)
    model = config.get("model_id", "gemini-2.5-flash-image-preview")

    tasks = []
    for filename in images_to_process:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        tasks.append((input_path, output_path, base_prompt, temperature, model, config))

    # print(tasks)

    if not config.get("keep_existing_banans", True):
        clear_output_directory(output_dir)
        # --- Run tasks in parallel ---
        num_workers = min(mp.cpu_count(), len(tasks))
        print(
            f"\nFound {len(tasks)} images. Starting parallel processing with {num_workers} workers..."
        )

        with mp.Pool(processes=num_workers) as pool:
            results = list(
                tqdm(pool.starmap(process_image_worker, tasks), total=len(tasks))
            )

        print("\n--- Processing Log ---")
        for res in results:
            print(res)
        print("----------------------")
        print("All tasks completed.")

    out_prepped = config["prepped_for_hunyuan"]
    # check if exist if not create
    if not os.path.exists(out_prepped):
        os.makedirs(out_prepped)

    # clean folder
    clear_output_directory(out_prepped)

    # Use 6x upscaling for even better edge quality with conservative background removal
    prepare_for_hunyuan(
        input_folder=output_dir,
        output=out_prepped,
        size=512,
        upscale_factor=2,
        conservative_mode=True,
    )

    # run empty room inpainting if configured
    empty_room_input = config["image_url"]
    empty_room_output = os.path.join(config["output_inp_banana"], "empty_room.png")

    prompt_empty_room = config.get(
        "prompt_empty_room",
        "Remove all objects and furniture from the room, making it empty."
    )

    if not config.get("keep_existing_empty_rooms", True):
        print("\nProcessing empty room image...")
        process_image_worker(
            input_image_path=empty_room_input,
            output_image_path=empty_room_output,
            base_prompt=prompt_empty_room,
            temperature=config.get("genai_temperature_emptyRoom", 0.05),
            config=config,
        )
    else:
        print("\nSkipping empty room processing as per configuration.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
