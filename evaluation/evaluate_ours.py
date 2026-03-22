from t2i_generation import t2i_generate_image

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from model import LLaDAForMultiModalGeneration
import argparse
import json
from PIL import Image
import re
import shutil
import ast

from transformers import AutoModel, AutoTokenizer
from sam2.sam2_image_predictor import SAM2ImagePredictor

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import argparse

import torchvision.transforms as T
from torchvision.transforms.functional import to_tensor, to_pil_image


def get_masking_args(src_img, input_prompt, output_dir, mask_map_path=""):
    args = argparse.Namespace(
                checkpoint="Alpha-VLLM/Lumina-DiMOO",
                prompt=input_prompt,
                src_img=src_img,
                timesteps=64,
                cfg_scale=4.0,
                temperature=1.0,
                seed=65513,
                vae_ckpt="Alpha-VLLM/Lumina-DiMOO",
                output_dir=output_dir,
                image_save_path=mask_map_path,
    )
    return args

def get_editing_args(src_img, input_prompt, inpainting_prompt_text, output_dir, image_save_path,
                     lamda1_ratio=0.2, if_w_o_refinement_segement=False,
                     if_w_o_specific_pipline=False, if_w_o_inpaint_lower_confidence=False,
                     if_w_o_inversion=False, target_subject_token=""):
    args = argparse.Namespace(
        checkpoint="Alpha-VLLM/Lumina-DiMOO",
        prompt=input_prompt,
        inpainting_prompt_text=inpainting_prompt_text,
        src_img=src_img,
        timesteps=64,
        cfg_scale=4.0,
        temperature=1.0,
        seed=65513,
        vae_ckpt="Alpha-VLLM/Lumina-DiMOO",
        output_dir=output_dir,
        image_save_path=image_save_path,
        lamda1_ratio=lamda1_ratio,
        if_w_o_refinement_segement=if_w_o_refinement_segement,
        if_w_o_specific_pipline=if_w_o_specific_pipline,
        if_w_o_inpaint_lower_confidence=if_w_o_inpaint_lower_confidence,
        if_w_o_inversion=if_w_o_inversion,
        target_subject_token=target_subject_token,
    )
    return args


def apply_mask_and_merge(image0, image1, mask1):
    img0 = to_tensor(image0).cpu()
    img1 = to_tensor(image1).cpu()
    mask = mask1
    mask = mask.float().cpu()

    while mask.dim() > 3:
        mask = mask.squeeze(0)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)    # → (1, H, W)
    # If mask is (1, H, W) or (H, W), broadcast to (3, H, W)
    if mask.shape[0] == 1:
        mask = mask.repeat(3, 1, 1)
    assert mask.shape == img0.shape, f"Mask shape {mask.shape} != image shape {img0.shape}"

    out_tensor = torch.where(mask.bool(), img1, img0)
    return to_pil_image(out_tensor)


def parse_args():
    parser = argparse.ArgumentParser(description="Lumina-DiMOO inference options")

    parser.add_argument(
        "--lambda1-ratio",
        type=float,
        default=0.2,
        help="Mixing coefficient lambda1 ratio"
    )

    # bool flags (default False)
    parser.add_argument(
        "--use-grounding-module",
        action="store_true",
        help="Enable grounding module"
    )

    parser.add_argument(
        "--w-o-refinement-segement",
        action="store_true",
        help="Disable refinement segmentation"
    )

    parser.add_argument(
        "--w-o-inversion",
        action="store_true",
        help="Disable inversion"
    )

    parser.add_argument(
        "--w-o-residual-recovery",
        action="store_true",
        help="Disable residual recovery"
    )

    parser.add_argument(
        "--w-o-intrinsic-refinement",
        action="store_true",
        help="Disable intrinsic refinement"
    )

    return parser.parse_args()

def parse_region_string(text: str) -> tuple:
    point_pattern = r"four-point region defined by the coordinates\s*(\[\[.*?\]\])"
    point_match = re.search(point_pattern, text)

    if point_match:
        try:
            coords_str = point_match.group(1)
            coords_list = ast.literal_eval(coords_str)
            return "point-based", coords_list
        except (SyntaxError, ValueError):
            pass

    box_pattern = r"the bounding box\s*(\[.*?\])"
    box_match = re.search(box_pattern, text)

    if box_match:
        try:
            coords_str = box_match.group(1)
            coords_list = ast.literal_eval(coords_str)
            return "box-based", coords_list
        except (SyntaxError, ValueError):
            pass

    return "text-only", []

def main():
    args = parse_args()
    lamda1_ratio = args.lambda1_ratio
    if_use_grounding_module = args.use_grounding_module
    if_w_o_refinement_segement = args.w_o_refinement_segement
    if_w_o_inversion = args.w_o_inversion
    if_w_o_residual_recovery = args.w_o_residual_recovery
    if_w_o_intrinsic_refinement = args.w_o_intrinsic_refinement

    # Load model and tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("Alpha-VLLM/Lumina-DiMOO", trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        "Alpha-VLLM/Lumina-DiMOO", torch_dtype=torch.bfloat16, device_map="auto",
    )
    # Load VQ-VAE
    from diffusers import VQModel
    vqvae = VQModel.from_pretrained("Alpha-VLLM/Lumina-DiMOO", subfolder="vqvae").to(device)
    sam2_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    sam3_model = build_sam3_image_model(
        checkpoint_path="your_sam3_model_dir/sam3.pt" # you need to download the sam3 model checkpoint and provide the path here
    )
    sam3_processor = Sam3Processor(sam3_model)
    image_base_path = "GIDE-Bench/img"

    # lamda1_ratio = 0.2
    # if_use_grounding_module = True
    # if_w_o_refinement_segement = False
    # if_w_o_inversion = False
    # if_w_o_residual_recovery = False
    # if_w_o_intrinsic_refinement = False

    if if_w_o_refinement_segement:
        saved_img_base_path = f"evaluation_result_ours_w_o_refinement_lamda1_{lamda1_ratio}"
    elif if_w_o_inversion:
        saved_img_base_path = f"evaluation_result_ours_w_o_inversion"
    elif if_w_o_residual_recovery:
        saved_img_base_path = f"evaluation_result_ours_w_o_residual_recovery_lamda1_{lamda1_ratio}"
    elif if_w_o_intrinsic_refinement:
        saved_img_base_path = f"evaluation_result_ours_w_o_intrinsic_refinement_lamda1_{lamda1_ratio}"
    elif not if_use_grounding_module:
        saved_img_base_path = f"evaluation_result_ours_w_o_grounding_module_lamda1_{lamda1_ratio}"
    else:
        saved_img_base_path = f"evaluation_result_ours_lamda1_{lamda1_ratio}"

    edit_instructions_path = "GIDE-Bench/edit_instructions.json"
    with open(edit_instructions_path, "r", encoding="utf-8") as f:
        edit_instructions = json.load(f)
    global_description_path = "GIDE-Bench/inputal_all.json"
    with open(global_description_path, "r", encoding="utf-8") as f:
        global_description = json.load(f)

    for img_index in range(805):
        edit_instruction = edit_instructions[str(img_index)]["edit_instructions"]
        input_modality_type, region_coordinates = parse_region_string(edit_instruction)
        item = global_description[str(img_index)]
        points_list = [[], []]
        bounding_box_list = [[], []]

        if input_modality_type == "point-based":
            mask_mode_list = ["sam2_point", "sam3_text"]
            points_list[0] = region_coordinates
        elif input_modality_type == "box-based":
            mask_mode_list = ["sam3_box", "sam3_text"]
            bounding_box_list[0] = region_coordinates
        else:
            mask_mode_list = ["sam3_text", "sam3_text"]

        image_path = os.path.join(image_base_path, f"{img_index}.jpg")
        edit_types = item["edit_type"].split('_', 1)
        src_prompt = item["source_prompt"]

        target_prompts = [item["target_prompt1"], item["target_prompt2"]]
        source_subject_tokens = [item["subject_token"]["instruction_1"]["source_subject_token"],
                                 item["subject_token"]["instruction_2"]["source_subject_token"]]
        target_subject_tokens = [item["subject_token"]["instruction_1"]["target_subject_token"],
                                 item["subject_token"]["instruction_2"]["target_subject_token"]]

        image_list = []
        saved_img_dir = os.path.join(saved_img_base_path, str(img_index))
        os.makedirs(saved_img_dir, exist_ok=True)
        last_saved_img_path = os.path.join(saved_img_dir, f"output_last.jpg")
        if os.path.exists(last_saved_img_path):
            continue

        for turn_index in range(2):
            if if_w_o_refinement_segement:
                item_output_cache_path = f"evaluation_result_ours_w_o_refinement_lamda1_{lamda1_ratio}_cache_{turn_index}"
            elif if_w_o_inversion:
                item_output_cache_path = f"evaluation_result_ours_w_o_inversion_cache_{turn_index}"
            elif if_w_o_residual_recovery:
                item_output_cache_path = f"evaluation_result_ours_w_o_residual_recovery_lamda1_{lamda1_ratio}_cache_{turn_index}"
            elif if_w_o_intrinsic_refinement:
                item_output_cache_path = f"evaluation_result_ours_w_o_intrinsic_refinement_lamda1_{lamda1_ratio}_cache_{turn_index}"
            else:
                item_output_cache_path = f"evaluation_result_ours_lamda1_{lamda1_ratio}_if_grounding_module_{if_use_grounding_module}_cache_{turn_index}"

            save_img_path = os.path.join(saved_img_dir, f"output_{turn_index}.jpg")
            edit_type = edit_types[turn_index]
            source_subject_token = source_subject_tokens[turn_index]
            target_subject_token = target_subject_tokens[turn_index]
            target_prompt = target_prompts[turn_index]

            if turn_index == 1:
                if not(edit_types[0] == "add" and edit_types[1] == "add"):
                    image_path = os.path.join(saved_img_dir, f"output_0.jpg")
                    src_prompt = target_prompts[0]
            src_img = Image.open(image_path)

            if (source_subject_token == target_subject_token) and (edit_type == "replace"):
                inpaint_ms_minus_mt = False
            else:
                inpaint_ms_minus_mt = True

            if if_use_grounding_module:
                if if_w_o_inversion:
                    img_editing_args = get_editing_args(src_img, src_prompt, target_prompt, item_output_cache_path,
                                                        save_img_path,
                                                        lamda1_ratio=lamda1_ratio,
                                                        if_w_o_refinement_segement=if_w_o_refinement_segement,
                                                        if_w_o_inversion=if_w_o_inversion,
                                                        target_subject_token=target_subject_token)
                    img_editing_args.inpaint_ms_minus_mt = inpaint_ms_minus_mt
                    image = t2i_generate_image(img_editing_args, tokenizer, model, vqvae, if_edit=True,
                                               if_inversion=False, device=device,
                                               transformer_processor=None,
                                               use_attention_control=True, sam3_processor=sam3_processor,
                                               subject_token=source_subject_token,
                                               edit_type=edit_type, sam2_predictor=sam2_predictor,
                                               selected_points=points_list[turn_index],
                                               bounding_box=bounding_box_list[turn_index],
                                               mask_mode=mask_mode_list[turn_index])
                else:
                    if edit_type != "remove":
                        mask_map_path = "masking_map.png"
                        mask_and_zt_args = get_masking_args(src_img, src_prompt, item_output_cache_path, mask_map_path)
                        latents_mask = \
                            t2i_generate_image(mask_and_zt_args, tokenizer, model, vqvae, if_edit=False, if_inversion=True, device=device,
                                           use_attention_control=True, sam3_processor=sam3_processor, subject_token=source_subject_token, edit_type=edit_type,
                                            sam2_predictor=sam2_predictor,
                                          selected_points=points_list[turn_index],
                                          bounding_box=bounding_box_list[turn_index],
                                          mask_mode=mask_mode_list[turn_index])
                        inpainting_prompt_text = target_prompt
                        img_editing_args = get_editing_args(src_img, target_prompt, inpainting_prompt_text, item_output_cache_path, save_img_path,
                                lamda1_ratio=lamda1_ratio,
                                if_w_o_refinement_segement=if_w_o_refinement_segement,
                                if_w_o_specific_pipline=if_w_o_residual_recovery,
                                if_w_o_inpaint_lower_confidence=if_w_o_intrinsic_refinement)
                        img_editing_args.inpaint_ms_minus_mt = inpaint_ms_minus_mt
                        image = t2i_generate_image(img_editing_args, tokenizer, model, vqvae, if_edit=True, if_inversion=False, device=device,
                                        transformer_processor=None,
                                        use_attention_control=True, sam3_processor=sam3_processor, subject_token=target_subject_token,
                                        latents_mask=latents_mask,
                                        edit_type=edit_type, sam2_predictor=sam2_predictor)
                    else:
                        img_editing_args = get_editing_args(src_img, src_prompt, target_prompt, item_output_cache_path, save_img_path,
                                lamda1_ratio=lamda1_ratio,
                                if_w_o_refinement_segement=if_w_o_refinement_segement,
                                if_w_o_specific_pipline=if_w_o_residual_recovery,
                                if_w_o_inpaint_lower_confidence=if_w_o_intrinsic_refinement)
                        image = t2i_generate_image(img_editing_args, tokenizer, model, vqvae, if_edit=True, if_inversion=False, device=device,
                                        transformer_processor=None,
                                        use_attention_control=True, sam3_processor=sam3_processor, subject_token=source_subject_token,
                                        edit_type=edit_type, sam2_predictor=sam2_predictor,
                                       selected_points=points_list[turn_index],
                                       bounding_box=bounding_box_list[turn_index],
                                       mask_mode=mask_mode_list[turn_index])
            else:
                if edit_type == "remove":
                    edit_type = "remove_ablation"
                mask_map_path = "masking_map.png"
                mask_and_zt_args = get_masking_args(src_img, src_prompt, item_output_cache_path, mask_map_path)
                latents_mask = \
                    t2i_generate_image(mask_and_zt_args, tokenizer, model, vqvae, if_edit=False, if_inversion=True,
                                       device=device,
                                       use_attention_control=False, sam3_processor=sam3_processor,
                                       subject_token=source_subject_token, edit_type=edit_type,
                                       sam2_predictor=sam2_predictor,
                                       selected_points=points_list[turn_index],
                                       bounding_box=bounding_box_list[turn_index],
                                       mask_mode=mask_mode_list[turn_index])
                inpainting_prompt_text = target_prompt
                img_editing_args = get_editing_args(src_img, target_prompt, inpainting_prompt_text,
                                                    item_output_cache_path, save_img_path, lamda1_ratio)
                img_editing_args.inpaint_ms_minus_mt = inpaint_ms_minus_mt
                image = t2i_generate_image(img_editing_args, tokenizer, model, vqvae, if_edit=True, if_inversion=False,
                                           device=device,
                                           transformer_processor=None,
                                           use_attention_control=False, sam3_processor=sam3_processor,
                                           subject_token=target_subject_token,
                                           latents_mask=latents_mask,
                                           edit_type=edit_type, sam2_predictor=sam2_predictor)

            image_list.append(image)
            if turn_index == 1:
                if edit_types[0] == "add" and edit_types[1] == "add":
                    subject_token = target_subject_token
                    query_img = image
                    entities = [t.strip() for t in subject_token.split("and")]
                    all_masks = []
                    for ent in entities:
                        inference_state = sam3_processor.set_image(query_img)
                        output = sam3_processor.set_text_prompt(state=inference_state, prompt=ent)
                        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
                        ent_mask = masks.any(dim=0, keepdim=True)
                        all_masks.append(ent_mask)
                    mask_latter = torch.stack(all_masks, dim=0).any(dim=0)

            if os.path.exists(item_output_cache_path):
                shutil.rmtree(item_output_cache_path)
                print(f"removed cache directory: {item_output_cache_path}", flush=True)

        if edit_types[0] == "add" and edit_types[1] == "add":
            image_out = apply_mask_and_merge(image_list[0], image_list[1], mask_latter)
        else:
            image_out = image_list[1]
        image_out.save(last_saved_img_path)


if __name__ == '__main__':
    main()
