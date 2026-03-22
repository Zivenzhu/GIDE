import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
import json

#Uniworld-2
pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
print("pipeline loaded")

pipeline.load_lora_weights(
    "chestnutlzj/Edit-R1-Qwen-Image-Edit-2509",
    adapter_name="lora",
)
pipeline.set_adapters(["lora"], adapter_weights=[1])

pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)

def edit_image(image_path, prompt, saved_path):
    image = Image.open(image_path)
    inputs = {
        "image": [image],
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 40,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }
    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]
        output_image.save(saved_path)

with open("../GIDE-Bench/edit_instructions.json", "r", encoding="utf-8") as f:
    edit_ins = json.load(f)

base_path = "../GIDE-Bench"
saved_dir = "evaluation_result_edit_r1"
os.makedirs(saved_dir, exist_ok=True)


for i in range(805):
    print(i)
    img_path = os.path.join(base_path, "img", f"{i}.jpg")
    prompt = edit_ins[str(i)]["edit_instructions"]
    saved_path = os.path.join(saved_dir, f"{i}.jpg")

    edit_image(img_path, prompt, saved_path)