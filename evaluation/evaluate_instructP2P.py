import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

import json
from PIL import Image, ImageOps

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
# `image` is an RGB PIL.Image


with open("../GIDE-Bench/edit_instructions.json", "r", encoding="utf-8") as f:
    edit_ins = json.load(f)

base_path = "../GIDE-Bench"
saved_dir = "evaluation_result_instructP2P"
os.makedirs(saved_dir, exist_ok=True)
for i in range(805):
    img_path = os.path.join(base_path, "img", f"{i}.jpg")
    image = Image.open(img_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    prompt = edit_ins[str(i)]["edit_instructions"]
    saved_path = os.path.join(saved_dir, f"{i}.jpg")

    out_images = pipe(prompt, image=image).images
    out_images[0].save(saved_path)