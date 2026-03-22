import base64
from openai import OpenAI
import os
import json

client = OpenAI(api_key="") # TODO: add your API key here
with open("../GIDE-Bench/edit_instructions.json", "r", encoding="utf-8") as f:
    edit_ins = json.load(f)

base_path = "../GIDE-Bench"
saved_dir = "evaluation_result_gpt"
os.makedirs(saved_dir, exist_ok=True)
for i in range(805):
    img_path = os.path.join(base_path, "img", f"{i}.jpg")
    prompt = edit_ins[str(i)]["edit_instructions"]
    saved_path = os.path.join(saved_dir, f"{i}.jpg")

    result = client.images.edit(
        model="gpt-image-1",
        image=[
            open(img_path, "rb"),
        ],
        prompt=prompt
    )

    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    # Save the image to a file
    with open(saved_path, "wb") as f:
        f.write(image_bytes)