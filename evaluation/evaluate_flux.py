import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import json


pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

def edit_image(image_path, prompt, saved_path):
    input_image = load_image(image_path)
    image = pipe(
      image=input_image,
      prompt=prompt,
      guidance_scale=2.5
    ).images[0]
    image.save(saved_path)


with open("../GIDE-Bench/edit_instructions.json", "r", encoding="utf-8") as f:
    edit_ins = json.load(f)

base_path = "../GIDE-Bench"
saved_dir = "evaluation_result_flux"
os.makedirs(saved_dir, exist_ok=True)


for i in range(805):
    print(i)
    img_path = os.path.join(base_path, "img", f"{i}.jpg")
    prompt = edit_ins[str(i)]["edit_instructions"]
    saved_path = os.path.join(saved_dir, f"{i}.jpg")

    edit_image(img_path, prompt, saved_path)