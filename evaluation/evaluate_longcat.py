import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import torch
from PIL import Image
from diffusers import LongCatImageEditPipeline

device = torch.device('cuda')

pipe = LongCatImageEditPipeline.from_pretrained("meituan-longcat/LongCat-Image-Edit", torch_dtype= torch.bfloat16 )
# pipe.to(device, torch.bfloat16)  # Uncomment for high VRAM devices (Faster inference)
pipe.enable_model_cpu_offload()  # Offload to CPU to save VRAM (Required ~18 GB); slower but prevents OOM

def edit_image(image_path, prompt, saved_path):
    img = Image.open(image_path).convert('RGB')
    image = pipe(
        img,
        prompt,
        negative_prompt='',
        guidance_scale=4.5,
        num_inference_steps=50,
        num_images_per_prompt=1,
        generator=torch.Generator("cpu").manual_seed(43)
    ).images[0]
    image.save(saved_path)



with open("../GIDE-Bench/edit_instructions.json", "r", encoding="utf-8") as f:
    edit_ins = json.load(f)

base_path = "../GIDE-Bench"
saved_dir = "evaluation_result_longcat"
os.makedirs(saved_dir, exist_ok=True)


for i in range(805):
    print(i)
    img_path = os.path.join(base_path, "img", f"{i}.jpg")
    prompt = edit_ins[str(i)]["edit_instructions"]
    saved_path = os.path.join(saved_dir, f"{i}.jpg")

    edit_image(img_path, prompt, saved_path)