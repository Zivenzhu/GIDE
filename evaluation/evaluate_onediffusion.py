import os
import json
import torch
from onediffusion.diffusion.pipelines.onediffusion import OneDiffusionPipeline
from PIL import Image

device = torch.device('cuda')
pipeline = OneDiffusionPipeline.from_pretrained("lehduong/OneDiffusion").to(device=device, dtype=torch.bfloat16)
NEGATIVE_PROMPT = "monochrome, greyscale, low-res, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation"

def generate_image(image_path, edit_prompt, save_path):
    images = [
        Image.open(image_path),
    ]
    prompt = f"[[image_editing]] {edit_prompt}"  # you can omit caption i.e., setting prompt to "[[depth2image]]"
    # set the denoise mask to [0, 1] to denoise condition (depth, pose, hed, canny, semantic map etc)
    # by default, the height and width will be set so that input image is minimally cropped
    ret = pipeline.img2img(
        image=images,
        num_inference_steps=50,
        prompt=prompt,
        denoise_mask=[0, 1],
        guidance_scale=4,
        NEGATIVE_PROMPT=NEGATIVE_PROMPT,
        # height=512,
        # width=512,
    )
    ret.images[0].save(save_path)


with open("../GIDE-Bench/edit_instructions.json", "r", encoding="utf-8") as f:
    edit_ins = json.load(f)

base_path = "../GIDE-Bench"
saved_dir = "evaluation_result_onediffusion"
os.makedirs(saved_dir, exist_ok=True)
for i in range(805):
    print(i)
    img_path = os.path.join(base_path, "img", f"{i}.jpg")
    prompt = edit_ins[str(i)]["edit_instructions"]
    saved_path = os.path.join(saved_dir, f"{i}.jpg")
    generate_image(img_path, prompt, saved_path)
