import torch
from diffusers.utils import load_image
from diffusers import (
    FluxControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionUpscalePipeline,
)
from diffusers.pipelines import FluxControlNetPipeline
from PIL import Image
import os


class Upscaler:
    def __init__(
        self, model_name: str = "SD", device: str = "cuda:0"
    ):  # "black-forest-labs/FLUX.1-dev"):
        if model_name == "SD":
            self.model_name = "stabilityai/stable-diffusion-x4-upscaler"

            self.pipe = StableDiffusionUpscalePipeline.from_pretrained(
                self.model_name, variant="fp16", torch_dtype=torch.float16
            )

            self.pipe = self.pipe.to(torch.device(device))

        elif model_name == "FLUX":
            self.model_name = "black-forest-labs/FLUX.1-dev"

            self.controlnet = FluxControlNetModel.from_pretrained(
                "jasperai/Flux.1-dev-Controlnet-Upscaler", torch_dtype=torch.float16
            )
            self.pipe = FluxControlNetPipeline.from_pretrained(
                self.model_name, controlnet=self.controlnet, torch_dtype=torch.float16
            )
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_attention_slicing()
            # self.pipe.enable_vae_tiling()
            # self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.safety_checker = None

        else:
            raise Exception(" Choose between 'SD' or 'FLUX' ")

    def __call__(
        self,
        control_image,
        size=128,
        num_inference_steps=5,
        guidance_scale=3.5,
        target_size=512,
    ):

        w, h = control_image.size

        # reize to correct square 512x512 px size without changing aspect ratio, add white padding if needed
        if w != h:
            if w > h:
                new_w = size
                new_h = int(h * (size / w))
            else:
                new_h = size
                new_w = int(w * (size / h))
        else:
            new_w = size
            new_h = size

        # Resize the image
        resized_image = control_image.resize((new_w, new_h), Image.LANCZOS)

        # Create a new white background image
        new_image = Image.new("RGB", (size, size), (255, 255, 255))

        # Paste resized image onto center of white background
        upper_left_x = (size - new_w) // 2
        upper_left_y = (size - new_h) // 2
        new_image.paste(resized_image, (upper_left_x, upper_left_y))

        control_image = new_image

        if self.model_name == "stabilityai/stable-diffusion-x4-upscaler":
            image = self.pipe(
                prompt="Upscale the furniture image",
                negative_prompt="low quality, blurry, pixelated, distorted",
                image=control_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]

        else:
            image = self.pipe(
                prompt="Upscale the furniture image",
                negative_prompt="low quality, blurry, pixelated, distorted",
                control_image=control_image,
                controlnet_conditioning_scale=1.0,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=control_image.size[1],
                width=control_image.size[0],
            ).images[0]

        # resize to 512x512 px for Hunyuan3D
        image = image.resize((target_size, target_size), Image.LANCZOS)
        # remove white background, making it alpha channel
        image = image.convert("RGBA")
        datas = image.getdata()
        newData = []
        for item in datas:
            # change all white (also shades of whites)
            # to transparent
            if item[0] > 240 and item[1] > 240 and item[2] > 240:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        image.putdata(newData)

        return image
