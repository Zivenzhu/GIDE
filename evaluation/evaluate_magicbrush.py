from PIL import Image, ImageOps
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import json
import os


class MagicBrush():
    def __init__(self, weight="vinesmsuic/magicbrush-jul7"):
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            weight,
            torch_dtype=torch.float16
        ).to("cuda")
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

    def infer_one_image(self, src_image, instruct_prompt, seed):
        generator = torch.manual_seed(seed)
        image = \
        self.pipe(instruct_prompt, image=src_image, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7,
                  generator=generator).images[0]
        return image


model = MagicBrush()


with open("../GIDE-Bench/edit_instructions.json", "r", encoding="utf-8") as f:
    edit_ins = json.load(f)

base_path = "../GIDE-Bench"
saved_dir = "evaluation_result_magicbrush"
os.makedirs(saved_dir, exist_ok=True)
for i in range(805):
    print(i)
    img_path = os.path.join(base_path, "img", f"{i}.jpg")
    image = Image.open(img_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    prompt = edit_ins[str(i)]["edit_instructions"]
    saved_path = os.path.join(saved_dir, f"{i}.jpg")

    image_output = model.infer_one_image(image, prompt, 42)
    image_output.save(saved_path)


