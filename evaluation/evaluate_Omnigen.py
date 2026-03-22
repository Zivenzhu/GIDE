import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from diffusers import OmniGenPipeline
from diffusers.utils import load_image
import os
import json

pipe = OmniGenPipeline.from_pretrained(
    "Shitao/OmniGen-v1-diffusers",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

def edit_image(image_path, edit_prompt, saved_path):
    prompt=f"<img><|image_1|></img> {edit_prompt}"
    input_images=[load_image(image_path)]
    image = pipe(
        prompt=prompt,
        input_images=input_images,
        guidance_scale=2,
        img_guidance_scale=1.6,
        use_input_image_size_as_output=True,
        generator=torch.Generator(device="cpu").manual_seed(222)
    ).images[0]
    image.save(saved_path)


with open("../GIDE-Bench/edit_instructions.json", "r", encoding="utf-8") as f:
    edit_ins = json.load(f)

base_path = "../GIDE-Bench"
saved_dir = "evaluation_result_omnigen"
os.makedirs(saved_dir, exist_ok=True)


for i in range(805):
    print(i)
    img_path = os.path.join(base_path, "img", f"{i}.jpg")
    prompt = edit_ins[str(i)]["edit_instructions"]
    saved_path = os.path.join(saved_dir, f"{i}.jpg")
    if os.path.exists(saved_path):
        continue

    edit_image(img_path, prompt, saved_path)