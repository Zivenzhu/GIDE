# -*- coding: utf-8 -*-
"""
Text-to-image inference script
"""
import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time
import torch
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from config import SPECIAL_TOKENS
from utils.generation_utils import setup_seed
from utils.image_utils import decode_vq_to_image, calculate_vq_params, add_break_line, \
    encode_img_with_breaks
from generators.image_generation_generator import generate_image, edit_image, get_inversion, generate_y0_logit, refine_image, inpaint_image, get_inversion_dice
from utils.prompt_utils import generate_text_to_image_prompt, create_prompt_templates
import numpy as np
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.visualization_utils import normalize_bbox


def get_image_tokens(vqvae, img):
    # Encode image
    input_img_token = encode_img_with_breaks(img, vqvae=vqvae)
    return input_img_token


def t2i_generate_image(args, tokenizer, model, vqvae, if_edit, if_inversion, device, transformer_processor=None,
                       use_attention_control=False, subject_token=None, sam3_processor=None,
                       latents_mask=None, edit_type=None, mask_mode="sam3_text", sam2_predictor=None,
                       selected_points=None, bounding_box=None):

    edit_type_raw = edit_type
    if if_edit and args.if_w_o_inversion:
        edit_type = "remove"


    img = args.src_img.convert("RGB")
    # crop_size_list = generate_crop_size_list((512 // 32) ** 2, 32)
    # img = var_center_crop(img, crop_size_list=crop_size_list)

    image_width, image_height = img.size
    args.height = image_height
    args.width = image_width

    # Special tokens
    MASK = SPECIAL_TOKENS["mask_token"]
    NEW_LINE = SPECIAL_TOKENS["newline_token"]
    BOA = SPECIAL_TOKENS["answer_start"]  # Begin of Answer
    EOA = SPECIAL_TOKENS["answer_end"]  # End of Answer
    BOI = SPECIAL_TOKENS["boi"]  # Begin of Image
    EOI = SPECIAL_TOKENS["eoi"]  # End of Image

    # Set Random seed
    if args.seed != 0:
        setup_seed(args.seed)

    # Create Output directory
    os.makedirs(args.output_dir, exist_ok=True)


    seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(args.height, args.width)

    args.token_grid_height = token_grid_height
    args.token_grid_width = token_grid_width
    # print(f"Generate image size: {args.height}x{args.width}")
    # print(f"Calculated VQ sequence length: {seq_len}")
    # print(f"Tokens per line (newline_every): {newline_every}")

    # Get prompt templates
    templates = create_prompt_templates()

    # Get prompt
    prompt_text = args.prompt

    # Generate prompts using utility function
    input_prompt, uncon_prompt = generate_text_to_image_prompt(prompt_text, templates)
    input_prompt_2, _ = generate_text_to_image_prompt(subject_token, templates)

    target_object_indices = []
    target_object_indices_2 = []

    if if_edit:
        inpainting_prompt_text = args.inpainting_prompt_text
        input_inpainting_prompt, _ = generate_text_to_image_prompt(inpainting_prompt_text, templates)
        inpaint_con_prompt_token = tokenizer(input_inpainting_prompt)["input_ids"]
        inpaint_con_prompt_ids = torch.tensor(inpaint_con_prompt_token, device=device).unsqueeze(0)


    if use_attention_control:
        enc = tokenizer(input_prompt, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        for i in range(len(offsets)):
            if offsets[i][0] is None:
                continue
            for j in range(i, len(offsets)):
                if offsets[j][1] is None:
                    break
                start = offsets[i][0]
                end = offsets[j][1]
                substring = input_prompt[start:end].lower()
                if substring.lower().strip() == subject_token.lower().strip():
                    for idx in range(i, j + 1):
                        target_object_indices.append(idx)
                    break
                if len(substring.lower().strip()) > len(subject_token.lower().strip()):
                    break

        enc_2 = tokenizer(input_prompt_2, return_offsets_mapping=True)
        offsets_2 = enc_2["offset_mapping"]
        for i in range(len(offsets_2)):
            if offsets_2[i][0] is None:
                continue
            for j in range(i, len(offsets_2)):
                if offsets_2[j][1] is None:
                    break
                start = offsets_2[i][0]
                end = offsets_2[j][1]
                substring = input_prompt_2[start:end].lower()
                if substring.lower().strip() == subject_token.lower().strip():
                    for idx in range(i, j + 1):
                        target_object_indices_2.append(idx)
                    break
                if len(substring.lower().strip()) > len(subject_token.lower().strip()):
                    break
        # assert len(target_object_indices_2) > 0

        if if_inversion:
            if mask_mode == "sam3_text":
                entities = [t.strip() for t in subject_token.split("and")]
                all_masks = []
                for ent in entities:
                    inference_state = sam3_processor.set_image(img)
                    output = sam3_processor.set_text_prompt(state=inference_state, prompt=ent)
                    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
                    ent_mask = masks.any(dim=0, keepdim=True)
                    all_masks.append(ent_mask)
                subject_mask = torch.stack(all_masks, dim=0).any(dim=0)

            elif mask_mode == "sam2_point":
                point_coords = np.array(selected_points)
                point_labels = np.ones(point_coords.shape[0], dtype=int)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    sam2_predictor.set_image(img)
                    masks, scores, logits = sam2_predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=False,
                    )
                subject_mask = torch.tensor(masks[0], device=device)

            elif mask_mode == "sam3_box":
                inference_state = sam3_processor.set_image(img)
                box_input_xywh = torch.tensor(bounding_box).view(-1, 4)
                box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)
                norm_box_cxcywh = normalize_bbox(box_input_cxcywh, args.width, args.height).flatten().tolist()
                sam3_processor.reset_all_prompts(inference_state)
                output = sam3_processor.add_geometric_prompt(
                    state=inference_state, box=norm_box_cxcywh, label=True
                )
                masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
                subject_mask = masks.any(dim=0, keepdim=True)

            else:
                raise NotImplementedError

            torch.save(subject_mask, os.path.join(args.output_dir, "object_mask.pt"))
            latents_mask = torch.nn.functional.interpolate(
                subject_mask.float().view(1, 1, subject_mask.shape[-2], subject_mask.shape[-1]),
                size=(token_grid_height, token_grid_width),
                mode='bilinear'
            ).view(seq_len).to(device)
            latents_mask = torch.clamp(latents_mask, max=0.01)

            if torch.all(latents_mask == 0):
                mask_mode = "attention_sam2"

        elif if_edit and edit_type == "remove":
            if mask_mode == "sam3_text":
                entities = [t.strip() for t in subject_token.split("and")]
                all_masks = []
                for ent in entities:
                    inference_state = sam3_processor.set_image(img)
                    output = sam3_processor.set_text_prompt(state=inference_state, prompt=ent)
                    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
                    ent_mask = masks.any(dim=0, keepdim=True)
                    all_masks.append(ent_mask)
                subject_mask = torch.stack(all_masks, dim=0).any(dim=0)

            elif mask_mode == "sam2_point":
                point_coords = np.array(selected_points)
                point_labels = np.ones(point_coords.shape[0], dtype=int)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    sam2_predictor.set_image(img)
                    masks, scores, logits = sam2_predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=False,
                    )
                subject_mask = torch.tensor(masks[0], device=device)

            elif mask_mode == "sam3_box":
                inference_state = sam3_processor.set_image(img)
                box_input_xywh = torch.tensor(bounding_box).view(-1, 4)
                box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)
                norm_box_cxcywh = normalize_bbox(box_input_cxcywh, args.width, args.height).flatten().tolist()
                sam3_processor.reset_all_prompts(inference_state)
                output = sam3_processor.add_geometric_prompt(
                    state=inference_state, box=norm_box_cxcywh, label=True
                )
                masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
                subject_mask = masks.any(dim=0, keepdim=True)

            else:
                raise NotImplementedError


            latents_mask = torch.nn.functional.interpolate(
                subject_mask.float().view(1, 1, subject_mask.shape[-2], subject_mask.shape[-1]),
                size=(token_grid_height, token_grid_width),
                mode='bilinear'
            ).view(token_grid_height, token_grid_width).to(device)
            latents_mask = torch.clamp(latents_mask, max=0.01)
            mask_inpaint = (latents_mask == 0.01).unsqueeze(0)

            if torch.all(latents_mask == 0):
                mask_mode = "attention_sam2"
                mask_inpaint = None
    else:
        latents_mask = torch.full((seq_len,), 0.01, device=device)


    # build initial sequence
    con_prompt_token = tokenizer(input_prompt)["input_ids"]
    uncon_prompt_token = tokenizer(uncon_prompt)["input_ids"]

    con_prompt_token_2 = tokenizer(input_prompt_2)["input_ids"]
    con_prompt_ids_2 = torch.tensor(con_prompt_token_2, device=device).unsqueeze(0)

    img_pred_token = [BOA] + get_image_tokens(vqvae, img) + [EOA]
    mask_begin_token = [BOA] + [BOI] + add_break_line([MASK] * seq_len, token_grid_height, token_grid_width,
                                                      new_number=NEW_LINE) + [EOI] + [EOA]
    mask_begin = torch.tensor(con_prompt_token + mask_begin_token, device=device).unsqueeze(0)
    mask_begin = mask_begin == MASK


    prompt_ids = torch.tensor(con_prompt_token + img_pred_token, device=device).unsqueeze(0)
    uncon_ids = torch.tensor(uncon_prompt_token, device=device).unsqueeze(0)

    # image satrt index
    code_start = len(con_prompt_token) + 2
    uncon_code_start = len(uncon_prompt_token) + 2

    args.vqvae = vqvae
    if not if_edit:
        start_time = time.time()
        if not if_inversion:
            # Generate VQ tokens
            vq_tokens = generate_image(
                model,
                prompt_ids,
                seq_len=seq_len,
                newline_every=newline_every,
                timesteps=args.timesteps,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                uncon_ids=uncon_ids,
                code_start=code_start,
                uncon_code_start=uncon_code_start,
                output_dir=args.output_dir
            )
        else:
            latents_mask = get_inversion(
                model,
                prompt_ids,
                mask_begin,

                args=args,
                mask_mode=mask_mode,
                sam2_predictor=sam2_predictor,
                image=img,
                target_object_indices=target_object_indices,
                con_prompt_ids_2=con_prompt_ids_2,
                target_object_indices_2=target_object_indices_2,

                use_attention_control=use_attention_control,
                latents_mask=latents_mask,
                seq_len=seq_len,
                newline_every=newline_every,
                timesteps=args.timesteps,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                uncon_ids=uncon_ids,
                code_start=code_start,
                uncon_code_start=uncon_code_start,
                output_dir=args.output_dir
            )
            return latents_mask
    else:
        # Generate VQ tokens
        start_time = time.time()
        if edit_type != "remove":
            vq_tokens, x = edit_image(
                model,
                prompt_ids,
                lamda1_ratio=args.lamda1_ratio,
                seq_len=seq_len,
                newline_every=newline_every,
                timesteps=args.timesteps,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                uncon_ids=uncon_ids,
                code_start=code_start,
                uncon_code_start=uncon_code_start,
                output_dir=args.output_dir,
                transformer_processor=transformer_processor,
            )

            new_path = args.image_save_path.replace(".jpg", "_after_inversion.jpg")
            _ = decode_vq_to_image(
                vq_tokens, new_path,
                vae_ckpt=args.vae_ckpt,
                image_height=args.height,
                image_width=args.width,
                vqvae=vqvae
            )

            args.subject_token = subject_token
            if args.if_w_o_refinement_segement:
                pass
            else:
                vq_tokens = refine_image(
                    model,
                    prompt=x,
                    args=args,
                    edit_type=edit_type,
                    raw_image_ids=torch.tensor(img_pred_token, device=device).unsqueeze(0),
                    # inpaint_con_prompt_ids=inpaint_con_prompt_ids,
                    sam3_processor=sam3_processor,

                    use_attention_control=use_attention_control,
                    sam2_predictor=sam2_predictor,
                    target_object_indices=target_object_indices,
                    con_prompt_ids_2=con_prompt_ids_2,
                    target_object_indices_2=target_object_indices_2,

                    mask_begin=mask_begin,
                    edited_img_tmp_ids=vq_tokens,
                    latents_mask_source=latents_mask,
                    seq_len=seq_len,
                    newline_every=newline_every,
                    timesteps=args.timesteps,
                    temperature=args.temperature,
                    cfg_scale=args.cfg_scale,
                    uncon_ids=uncon_ids,
                    code_start=code_start,
                    uncon_code_start=uncon_code_start,
                )
        else:
            vq_tokens, x = inpaint_image(
                model,
                prompt=prompt_ids,
                args=args,
                inpaint_con_prompt_ids=inpaint_con_prompt_ids,

                mask_mode=mask_mode,
                sam2_predictor=sam2_predictor,
                image=img,
                target_object_indices=target_object_indices,
                con_prompt_ids_2=con_prompt_ids_2,
                target_object_indices_2=target_object_indices_2,

                mask_inpaint=mask_inpaint,
                mask_begin=mask_begin,
                seq_len=seq_len,
                newline_every=newline_every,
                timesteps=args.timesteps,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                uncon_ids=uncon_ids,
                code_start=code_start,
                uncon_code_start=uncon_code_start,
            )

            if edit_type_raw != "remove":
                prompt_text = args.inpainting_prompt_text
                subject_token = args.target_subject_token
                latents_mask = latents_mask.view(seq_len)
                args.subject_token = subject_token
                edit_type = edit_type_raw

                input_prompt, uncon_prompt = generate_text_to_image_prompt(prompt_text, templates)
                input_prompt_2, _ = generate_text_to_image_prompt(subject_token, templates)
                target_object_indices = []
                target_object_indices_2 = []

                if use_attention_control:
                    enc = tokenizer(input_prompt, return_offsets_mapping=True)
                    offsets = enc["offset_mapping"]
                    for i in range(len(offsets)):
                        if offsets[i][0] is None:
                            continue
                        for j in range(i, len(offsets)):
                            if offsets[j][1] is None:
                                break
                            start = offsets[i][0]
                            end = offsets[j][1]
                            substring = input_prompt[start:end].lower()
                            if substring.lower().strip() == subject_token.lower().strip():
                                for idx in range(i, j + 1):
                                    target_object_indices.append(idx)
                                break
                            if len(substring.lower().strip()) > len(subject_token.lower().strip()):
                                break

                    enc_2 = tokenizer(input_prompt_2, return_offsets_mapping=True)
                    offsets_2 = enc_2["offset_mapping"]
                    for i in range(len(offsets_2)):
                        if offsets_2[i][0] is None:
                            continue
                        for j in range(i, len(offsets_2)):
                            if offsets_2[j][1] is None:
                                break
                            start = offsets_2[i][0]
                            end = offsets_2[j][1]
                            substring = input_prompt_2[start:end].lower()
                            if substring.lower().strip() == subject_token.lower().strip():
                                for idx in range(i, j + 1):
                                    target_object_indices_2.append(idx)
                                break
                            if len(substring.lower().strip()) > len(subject_token.lower().strip()):
                                break
                    assert len(target_object_indices_2) > 0

                # build initial sequence
                con_prompt_token = tokenizer(input_prompt)["input_ids"]
                uncon_prompt_token = tokenizer(uncon_prompt)["input_ids"]

                con_prompt_token_2 = tokenizer(input_prompt_2)["input_ids"]
                con_prompt_ids_2 = torch.tensor(con_prompt_token_2, device=device).unsqueeze(0)

                img_pred_token = [BOA] + get_image_tokens(vqvae, img) + [EOA]
                mask_begin_token = [BOA] + [BOI] + add_break_line([MASK] * seq_len, token_grid_height, token_grid_width,
                                                                  new_number=NEW_LINE) + [EOI] + [EOA]
                mask_begin = torch.tensor(con_prompt_token + mask_begin_token, device=device).unsqueeze(0)
                mask_begin = mask_begin == MASK

                prompt_ids = torch.tensor(con_prompt_token + img_pred_token, device=device).unsqueeze(0)
                uncon_ids = torch.tensor(uncon_prompt_token, device=device).unsqueeze(0)

                # image satrt index
                code_start = len(con_prompt_token) + 2
                uncon_code_start = len(uncon_prompt_token) + 2

                vq_tokens = refine_image(
                    model,
                    prompt=x,
                    args=args,
                    edit_type=edit_type,
                    raw_image_ids=torch.tensor(img_pred_token, device=device).unsqueeze(0),
                    sam3_processor=sam3_processor,
                    use_attention_control=use_attention_control,
                    sam2_predictor=sam2_predictor,

                    target_object_indices=target_object_indices,
                    con_prompt_ids_2=con_prompt_ids_2,
                    target_object_indices_2=target_object_indices_2,
                    mask_begin=mask_begin,
                    edited_img_tmp_ids=vq_tokens,
                    latents_mask_source=latents_mask,
                    seq_len=seq_len,
                    newline_every=newline_every,
                    timesteps=args.timesteps,
                    temperature=args.temperature,
                    cfg_scale=args.cfg_scale,
                    uncon_ids=uncon_ids,
                    code_start=code_start,
                    uncon_code_start=uncon_code_start,
                )

    os.makedirs(os.path.dirname(args.image_save_path), exist_ok=True)

    # Decode VQ codes to PNG and save
    image = decode_vq_to_image(
        vq_tokens, args.image_save_path,
        vae_ckpt=args.vae_ckpt,
        image_height=args.height,
        image_width=args.width,
        vqvae=vqvae
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    # print(f"Time: {elapsed_time:.2f}s")
    return image


def dice_generate_image(args, tokenizer, model, vqvae, if_edit, if_inversion, device):
    img = args.src_img.convert("RGB")

    image_width, image_height = img.size
    args.height = image_height
    args.width = image_width


    MASK = SPECIAL_TOKENS["mask_token"]
    NEW_LINE = SPECIAL_TOKENS["newline_token"]
    BOA = SPECIAL_TOKENS["answer_start"]  # Begin of Answer
    EOA = SPECIAL_TOKENS["answer_end"]  # End of Answer
    BOI = SPECIAL_TOKENS["boi"]  # Begin of Image
    EOI = SPECIAL_TOKENS["eoi"]  # End of Image

    # Set Random seed
    if args.seed != 0:
        setup_seed(args.seed)

    # Create Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(args.height, args.width)

    templates = create_prompt_templates()

    prompt_text = args.prompt

    input_prompt, uncon_prompt = generate_text_to_image_prompt(prompt_text, templates)

    con_prompt_token = tokenizer(input_prompt)["input_ids"]
    uncon_prompt_token = tokenizer(uncon_prompt)["input_ids"]


    img_pred_token = [BOA] + get_image_tokens(vqvae, img) + [EOA]
    mask_begin_token = [BOA] + [BOI] + add_break_line([MASK] * seq_len, token_grid_height, token_grid_width,
                                                      new_number=NEW_LINE) + [EOI] + [EOA]
    mask_begin = torch.tensor(con_prompt_token + mask_begin_token, device=device).unsqueeze(0)
    mask_begin = mask_begin == MASK


    prompt_ids = torch.tensor(con_prompt_token + img_pred_token, device=device).unsqueeze(0)
    uncon_ids = torch.tensor(uncon_prompt_token, device=device).unsqueeze(0)

    # image satrt index
    code_start = len(con_prompt_token) + 2
    uncon_code_start = len(uncon_prompt_token) + 2

    args.vqvae = vqvae
    if not if_edit:
        y0_logit = generate_y0_logit(
            model,
            prompt_ids,
            seq_len=seq_len,
            cfg_scale=args.cfg_scale,
            uncon_ids=uncon_ids,
            code_start=code_start,
            uncon_code_start=uncon_code_start
        )
        torch.save(y0_logit, os.path.join(args.output_dir, "y0_logit.pt"))

        start_time = time.time()
        if not if_inversion:
            # Generate VQ tokens
            vq_tokens = generate_image(
                model,
                prompt_ids,
                seq_len=seq_len,
                newline_every=newline_every,
                timesteps=args.timesteps,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                uncon_ids=uncon_ids,
                code_start=code_start,
                uncon_code_start=uncon_code_start,
                output_dir=args.output_dir
            )
        else:
            get_inversion_dice(
                model,
                prompt_ids,
                mask_begin,
                seq_len=seq_len,
                newline_every=newline_every,
                timesteps=args.timesteps,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                uncon_ids=uncon_ids,
                code_start=code_start,
                uncon_code_start=uncon_code_start,
                output_dir=args.output_dir
            )
            return
            # print("inversion completed")
    else:
        start_time = time.time()
        vq_tokens, x = edit_image(
            model,
            prompt_ids,
            if_dice=True,
            seq_len=seq_len,
            newline_every=newline_every,
            timesteps=args.timesteps,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            uncon_ids=uncon_ids,
            code_start=code_start,
            uncon_code_start=uncon_code_start,
            output_dir=args.output_dir,
        )

    os.makedirs(os.path.dirname(args.image_save_path), exist_ok=True)

    # Decode VQ codes to PNG and save
    image = decode_vq_to_image(
        vq_tokens, args.image_save_path,
        vae_ckpt=args.vae_ckpt,
        image_height=args.height,
        image_width=args.width,
        vqvae=vqvae
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    # print(f"Time: {elapsed_time:.2f}s")
    return image

