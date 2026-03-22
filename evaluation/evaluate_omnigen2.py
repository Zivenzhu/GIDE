import subprocess
import os
import json

def run_omnigen_edit(
    input_image_path,
    instruction,
    output_image_path,
    model_path="OmniGen2/OmniGen2",
    num_inference_step=50,
    text_guidance_scale=5.0,
    image_guidance_scale=2.0,
    num_images_per_prompt=1,
):
    cmd = [
        "python", "OmniGen2/inference.py", # you may need to dowload OmniGen2 and change the path to inference.py
        "--model_path", model_path,
        "--num_inference_step", str(num_inference_step),
        "--text_guidance_scale", str(text_guidance_scale),
        "--image_guidance_scale", str(image_guidance_scale),
        "--instruction", instruction,
        "--input_image_path", input_image_path,
        "--output_image_path", output_image_path,
        "--num_images_per_prompt", str(num_images_per_prompt),
    ]

    subprocess.run(cmd, check=True)


with open("../GIDE-Bench/edit_instructions.json", "r", encoding="utf-8") as f:
    edit_ins = json.load(f)

base_path = "../GIDE-Bench"
saved_dir = "evaluation_result_omnigen2"
os.makedirs(saved_dir, exist_ok=True)
for i in range(805):
    print(i)
    img_path = os.path.join(base_path, "img", f"{i}.jpg")
    prompt = edit_ins[str(i)]["edit_instructions"]
    saved_path = os.path.join(saved_dir, f"{i}.jpg")

    run_omnigen_edit(img_path, prompt, saved_path)