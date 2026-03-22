import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from Logit_train.i2i_generation import i2i_generate_image
import torch
# from transformers import AutoConfig, AutoTokenizer
from model import LLaDAForMultiModalGeneration
import argparse
import json
from PIL import Image
import re
import shutil

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import AutoModel, AutoTokenizer


def get_i2i_args(src_img, input_prompt, edit_type, output_dir, image_save_path):
    args = argparse.Namespace(
                checkpoint="Alpha-VLLM/Lumina-DiMOO",
                prompt=input_prompt,
                src_img=src_img,
                edit_type=f"edit_{edit_type}",
                timesteps=64,
                cfg_scale=2.5,
                cfg_img=4.0,
                temperature=1.0,
                seed=65513,
                vae_ckpt="Alpha-VLLM/Lumina-DiMOO",
                output_dir=output_dir,
                image_save_path=image_save_path,
    )
    return args



def split_edit_instructions(instruction, edit_types):
    assert len(edit_types) == 2, "edit_types must have length 2"
    results = []
    text = instruction.strip()

    for i, edit_type in enumerate(edit_types):
        verb = edit_type.lower()
        if i < len(edit_types) - 1:
            next_verbs = "|".join(edit_types[i + 1:])
            pattern = rf'({verb}\b.*?)(?=(?:,?\s+and\s+)?(?:{next_verbs})\b)'
        else:
            pattern = rf'({verb}\b.*$)'

        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            raise ValueError(f"Cannot match instruction for edit type: {edit_type}")
        cmd = match.group(1).strip().rstrip(",")
        cmd = cmd[0].upper() + cmd[1:]
        if not cmd.endswith("."):
            cmd += "."
        results.append(cmd)
        text = text[match.end():]

    assert len(results) == 2
    return results

def main():
    # Load model and tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("Alpha-VLLM/Lumina-DiMOO", trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        "Alpha-VLLM/Lumina-DiMOO", torch_dtype=torch.bfloat16, device_map="auto",
    )
    # Load VQ-VAE
    from diffusers import VQModel
    vqvae = VQModel.from_pretrained("Alpha-VLLM/Lumina-DiMOO", subfolder="vqvae").to(device)


    image_base_path = "GIDE-Bench/img" 
    edit_ins_path = f"GIDE-Bench/edit_instructions.json"
    saved_img_base_path = f"evaluate_lumina"
    with open(edit_ins_path, "r", encoding="utf-8") as f:
        all_edit_ins = json.load(f)

    indexs = sorted([int(key) for key in all_edit_ins.keys()])
    for img_index in indexs:
        image_path = os.path.join(image_base_path, f"{img_index}.jpg")
        item = all_edit_ins[str(img_index)]
        edit_types = item["edit_type"].split('_', 1)
        all_ins = item["edit_instructions"]
        edit_inses = split_edit_instructions(all_ins, edit_types)

        saved_img_dir = os.path.join(saved_img_base_path, str(img_index))
        os.makedirs(saved_img_dir, exist_ok=True)
        last_saved_img_path = os.path.join(saved_img_dir, f"output_1.jpg")
        if os.path.exists(last_saved_img_path):
            continue

        for turn_index in range(2):
            item_output_cache_path = f"lumina_cache_{turn_index}"
            save_img_path = os.path.join(saved_img_dir, f"output_{turn_index}.jpg")
            edit_type = edit_types[turn_index]
            edit_ins = edit_inses[turn_index]

            if turn_index == 1:
                image_path = os.path.join(saved_img_dir, f"output_0.jpg")
            src_img = Image.open(image_path)

            img_editing_args = get_i2i_args(src_img, edit_ins, edit_type, item_output_cache_path, save_img_path)
            i2i_generate_image(img_editing_args, tokenizer, model, vqvae)

            if os.path.exists(item_output_cache_path):
                shutil.rmtree(item_output_cache_path)
                print(f"removed cache directory: {item_output_cache_path}")


if __name__ == '__main__':
    main()
