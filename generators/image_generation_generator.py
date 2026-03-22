# -*- coding: utf-8 -*-
"""
Image generation generator
"""
import torch
import math
from typing import Callable, Optional
from utils.generation_utils import cosine_schedule, gumbel_max_sample, mask_by_random_topk, sine_schedule, mask_by_random_topk_inversion
import os
import numpy as np
import torch.nn.functional as F
import re
from utils.image_utils import decode_vq_to_image


@torch.no_grad()
def generate_y0_logit(
    model,
    prompt: torch.LongTensor,
    *,
    seq_len: int = 1024,
    cfg_scale: float = 0.0,
    newline_id: int = 126084,
    uncon_ids: torch.LongTensor,
    mask_token_id: int = 126336,
    code_start: Optional[int] = None,
    uncon_code_start: Optional[int] = None,
    codebook_size: int = 8192,
    text_vocab_size: Optional[int] = None,
) -> torch.LongTensor:

    device = next(model.parameters()).device
    prompt = prompt.to(device)
    B, P = prompt.shape
    assert B == 1, "batch>1 not supported – wrap in loop if needed"

    x = prompt

    # Infer text vocabulary size
    if text_vocab_size is None:
        vocab_total = model(torch.zeros(1, 1, dtype=torch.long, device=device), infer=True).logits.size(-1)
        text_vocab_size = vocab_total - codebook_size
    vocab_offset = text_vocab_size

    # Forward pass (with/without CFG)
    if cfg_scale > 0:
        uncond = torch.cat((uncon_ids.to(x.device), x[:, code_start - 2:]), axis=1)
        cond_logits = model(x, infer=True).logits[:, code_start - 2:, vocab_offset: vocab_offset + codebook_size]
        uncond_logits = model(uncond, infer=True).logits[:, uncon_code_start - 2:,
                        vocab_offset: vocab_offset + codebook_size]
        logits = (1 + cfg_scale) * cond_logits - cfg_scale * uncond_logits
    else:
        logits = model(x, infer=True).logits[:, code_start - 2:, vocab_offset: vocab_offset + codebook_size]

    return logits

def find_min_step(output_dir):
    step_list = []
    pattern = re.compile(r"mask_step_(\d+)\.pt")

    for fname in os.listdir(output_dir):
        m = pattern.match(fname)
        if m:
            step_list.append(int(m.group(1)))

    if not step_list:
        return 0

    return min(step_list)

@torch.no_grad()
def edit_image(
    model,
    prompt: torch.LongTensor,
    lamda1_ratio,
    *,
    if_dice: bool=False,
    seq_len: int = 1024,
    newline_every: int = 16,
    timesteps: int = 18,
    mask_token_id: int = 126336,
    newline_id: int = 126084,
    temperature: float = 1.0,
    cfg_scale: float = 0.0,
    uncon_ids: torch.LongTensor,
    code_start: Optional[int] = None,
    uncon_code_start: Optional[int] = None,
    codebook_size: int = 8192,
    noise_schedule: Callable[[torch.Tensor], torch.Tensor] = cosine_schedule,
    text_vocab_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    output_dir: str = '',
    transformer_processor=None,
) -> torch.LongTensor:
    """
    MaskGit parallel decoding to generate VQ tokens

    Args:
        model: Model
        prompt: Prompt tensor
        seq_len: Sequence length
        newline_every: Newline interval per row
        timesteps: Number of timesteps
        mask_token_id: Mask token id
        newline_id: Newline token id
        temperature: Temperature
        cfg_scale: CFG scale
        uncon_ids: Unconditional input
        code_start: Image token satrt index
        codebook_size: Codebook size
        noise_schedule: Noise schedule function
        text_vocab_size: Text vocabulary size
        generator: Random number generator

    Returns:
        Final VQ codes (1, seq_len)
    """

    device = next(model.parameters()).device
    prompt = prompt.to(device)
    B, P = prompt.shape
    assert B == 1, "batch>1 not supported – wrap in loop if needed"

    x = prompt
    vq_mask = x == mask_token_id
    mask_initial = None

    # Infer text vocabulary size
    if text_vocab_size is None:
        vocab_total = model(torch.zeros(1, 1, dtype=torch.long, device=device), infer=True).logits.size(-1)
        text_vocab_size = vocab_total - codebook_size
    vocab_offset = text_vocab_size

    start_step = find_min_step(output_dir)
    for step in range(start_step, timesteps):
        mask_now = torch.load(os.path.join(output_dir, f"mask_step_{step}.pt"))
        vq_mask[:, code_start - 2:] = mask_now
        x[vq_mask] = mask_token_id

        unknown_cnt = vq_mask.sum(dim=1, keepdim=True)
        if unknown_cnt.item() == 0:
            break

        z_hidden_t = None

        # Forward pass (with/without CFG)
        if cfg_scale > 0:
            uncond = torch.cat((uncon_ids.to(x.device), x[:, code_start-2:]), axis=1)
            uncond_vq_mask = torch.cat((torch.zeros((1, uncon_ids.size()[1]), dtype=torch.bool).to(x.device), vq_mask[:, code_start-2:]), axis=1)
            cond_logits_raw = model(x, infer=True, added_hidden_state=z_hidden_t, mask_initial=mask_initial).logits
            cond_logits = cond_logits_raw[:, vq_mask[0], vocab_offset : vocab_offset + codebook_size]
            uncond_logits_raw = model(uncond, infer=True).logits
            uncond_logits = uncond_logits_raw[:, uncond_vq_mask[0], vocab_offset : vocab_offset + codebook_size]
            logits = (1 + cfg_scale) * cond_logits - cfg_scale * uncond_logits

        else:
            logits_raw = model(x, infer=True, added_hidden_state=z_hidden_t, mask_initial=mask_initial).logits
            logits = logits_raw[:, vq_mask[0], vocab_offset : vocab_offset + codebook_size]

        _, logits_after_noise = gumbel_max_sample(logits, temperature, generator=generator)
    
        zt = torch.load(os.path.join(output_dir, f"z_step_{step}.pt"))
        gumbel_dist = torch.distributions.gumbel.Gumbel(0.0, 1.0)
        g = gumbel_dist.sample(zt.shape).to(logits.device)

        #Ours: alpha = 0.2
        #Dice: alpha = 0.3

        alpha = lamda1_ratio
        logits_after_noise += alpha * zt + (1 - alpha) * g

        if transformer_processor is not None:
            logits_after_noise = transformer_processor(logits_after_noise)

        sampled = logits_after_noise.argmax(dim=-1)
        sampled_full = sampled + vocab_offset
        flat_idx = vq_mask.nonzero(as_tuple=False)[:, 1]
        x.view(-1)[flat_idx] = sampled_full.view(-1)

    # Remove newline tokens
    vq_ids = x[0, code_start:-2]
    vq_ids = vq_ids[vq_ids != newline_id].view(1, seq_len)

    return vq_ids, x

@torch.no_grad()
def refine_image(
    model,
    prompt: torch.LongTensor,
    args,
    edit_type,
    raw_image_ids,

    use_attention_control,
    sam2_predictor,
    target_object_indices,
    con_prompt_ids_2,
    target_object_indices_2,

    sam3_processor,
    mask_begin: torch.LongTensor,
    edited_img_tmp_ids,
    *,
    latents_mask_source=None,
    seq_len: int = 1024,
    newline_every: int = 16,
    timesteps: int = 18,
    mask_token_id: int = 126336,
    newline_id: int = 126084,
    temperature: float = 1.0,
    cfg_scale: float = 0.0,
    uncon_ids: torch.LongTensor=None,
    code_start: Optional[int] = None,
    uncon_code_start: Optional[int] = None,
    codebook_size: int = 8192,
    noise_schedule: Callable[[torch.Tensor], torch.Tensor] = cosine_schedule,
    text_vocab_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.LongTensor:

    device = next(model.parameters()).device
    prompt = prompt.to(device)
    B, P = prompt.shape
    assert B == 1, "batch>1 not supported – wrap in loop if needed"

    x = prompt
    vq_mask = mask_begin.clone()

    # Infer text vocabulary size
    if text_vocab_size is None:
        vocab_total = model(torch.zeros(1, 1, dtype=torch.long, device=device), infer=True).logits.size(-1)
        text_vocab_size = vocab_total - codebook_size
    vocab_offset = text_vocab_size

    z_hidden_t = None

    if args.if_w_o_inpaint_lower_confidence:
        pass
    else:
        if cfg_scale > 0:
            uncond = torch.cat((uncon_ids.to(x.device), x[:, code_start-2:]), axis=1)
            uncond_vq_mask = torch.cat((torch.zeros((1, uncon_ids.size()[1]), dtype=torch.bool).to(x.device), vq_mask[:, code_start-2:]), axis=1)
            cond_logits_raw = model(x, infer=True, added_hidden_state=z_hidden_t, mask_initial=mask_begin).logits
            cond_logits = cond_logits_raw[:, vq_mask[0], vocab_offset : vocab_offset + codebook_size]
            uncond_logits_raw = model(uncond, infer=True).logits
            uncond_logits = uncond_logits_raw[:, uncond_vq_mask[0], vocab_offset : vocab_offset + codebook_size]
            logits = (1 + cfg_scale) * cond_logits - cfg_scale * uncond_logits

        else:
            logits_raw = model(x, infer=True, added_hidden_state=z_hidden_t, mask_initial=mask_begin).logits
            logits = logits_raw[:, vq_mask[0], vocab_offset : vocab_offset + codebook_size]

        gt = edited_img_tmp_ids.unsqueeze(-1) - vocab_offset  # (1, seq_len, 1)
        selected_logits = torch.gather(logits, dim=-1, index=gt)  # (1, seq_len, 1)
        selected_logits = selected_logits.squeeze(-1)

        threshold = selected_logits.max() * 0.5
        mask_tmp = selected_logits < threshold
        mask_tmp = mask_tmp & (latents_mask_source.unsqueeze(0) == 0.01)

        # #Visualize confidence map
        # torch.save(mask_tmp, os.path.join(args.output_dir, f"lower_confidence_map.pt"))

        x[:, mask_begin[0]] = torch.where(mask_tmp, mask_token_id, x[:, mask_begin[0]])
        vq_mask = x == mask_token_id
        unknown_cnt = vq_mask.sum(dim=1, keepdim=True)
        vq_len = unknown_cnt

        #refine the area in the Mask Target
        for step in range(timesteps):
            if unknown_cnt.item() == 0:
                break

            # Calculate number of tokens to keep (continue masking) this round
            if step < timesteps - 1:
                frac = noise_schedule(torch.tensor([(step + 1) / timesteps], device=device))
                keep_n = (vq_len.float() * frac).floor().clamp_min(1).long()
            else:
                keep_n = torch.zeros_like(unknown_cnt)

            # Forward pass (with/without CFG)
            if cfg_scale > 0:
                uncond = torch.cat((uncon_ids.to(x.device), x[:, code_start-2:]), axis=1)
                uncond_vq_mask = torch.cat((torch.zeros((1, uncon_ids.size()[1]), dtype=torch.bool).to(x.device), vq_mask[:, code_start-2:]), axis=1)
                cond_logits_raw = model(x, infer=True).logits
                cond_logits = cond_logits_raw[:, vq_mask[0], vocab_offset : vocab_offset + codebook_size]
                uncond_logits_raw = model(uncond, infer=True).logits
                uncond_logits = uncond_logits_raw[:, uncond_vq_mask[0], vocab_offset : vocab_offset + codebook_size]
                logits = (1 + cfg_scale) * cond_logits - cfg_scale * uncond_logits
            else:
                logits_raw = model(x, infer=True).logits
                logits = logits_raw[:, vq_mask[0], vocab_offset : vocab_offset + codebook_size]

            sampled, y0_t_logit = gumbel_max_sample(logits, temperature, generator=generator)
            sampled_full = sampled + vocab_offset
            probs = torch.softmax(logits, dim=-1)
            conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

            flat_idx = vq_mask.nonzero(as_tuple=False)[:, 1]
            x.view(-1)[flat_idx] = sampled_full.view(-1)

            conf_map = torch.full_like(x, -math.inf, dtype=probs.dtype)
            conf_map.view(-1)[flat_idx] = conf.view(-1)

            mask_sel = mask_by_random_topk(keep_n.squeeze(1), conf, temperature=temperature, generator=generator)
            x.view(-1)[flat_idx[mask_sel.view(-1)]] = mask_token_id
            vq_mask = x == mask_token_id
            unknown_cnt = vq_mask.sum(dim=1, keepdim=True)

    after_inpaint_path = args.image_save_path.replace(".jpg", "_after_intrinsic_refinement.jpg")
    tmp_vq_ids = x[0, code_start:-2]
    tmp_vq_ids = tmp_vq_ids[tmp_vq_ids != newline_id].view(1, seq_len)
    image = decode_vq_to_image(
        tmp_vq_ids, after_inpaint_path,
        vae_ckpt=args.vae_ckpt,
        image_height=args.height,
        image_width=args.width,
        vqvae=args.vqvae
    )

    if args.if_w_o_specific_pipline:
        return tmp_vq_ids

    if use_attention_control:
        entities = [t.strip() for t in args.subject_token.split("and")]
        all_masks = []
        for ent in entities:
            inference_state = sam3_processor.set_image(image)
            output = sam3_processor.set_text_prompt(state=inference_state, prompt=ent)
            masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
            ent_mask = masks.any(dim=0, keepdim=True)
            all_masks.append(ent_mask)
        subject_mask = torch.stack(all_masks, dim=0).any(dim=0)

        if torch.all(subject_mask == 0):
            att_score_list = AttentionScore()
            att_score_list.modify_state(True)
            if target_object_indices == []:
                target_object_indices_used = target_object_indices_2
                input_x = torch.cat((con_prompt_ids_2, x[:, code_start - 2:]), axis=1)
                mask_2 = torch.zeros(
                    (mask_begin.shape[0], con_prompt_ids_2.shape[1]),
                    dtype=torch.bool,
                    device=mask_begin.device
                )
                mask_begin_used = torch.cat((mask_2, mask_begin[:, code_start - 2:]), axis=1)
            else:
                input_x = x.clone()
                target_object_indices_used = target_object_indices
                mask_begin_used = mask_begin.clone()

            _ = model(input_x, infer=True,
                    use_attention_control=True,
                    target_object_indices=target_object_indices_used,
                    image_mask=mask_begin_used.squeeze(0),
                    att_score_list=att_score_list)
            avg_attn = att_score_list.get_avg_att()
            avg_attn = avg_attn - avg_attn.min()
            threshold = avg_attn.max() * 0.8
            subject_mask, point_coords = attention_to_points_sam_predict(avg_attn,
                                                                         (avg_attn > threshold).to(avg_attn.dtype),
                                                                         sam2_predictor, image, device)
        torch.save(subject_mask, os.path.join(args.output_dir, "target_object_mask.pt"))

        latents_mask_target = torch.nn.functional.interpolate(
            subject_mask.float().view(1, 1, subject_mask.shape[-2], subject_mask.shape[-1]),
            size=(subject_mask.shape[-2] // 16, subject_mask.shape[-1] // 16),
            mode='bilinear'
        ).view(seq_len).to(device)
        latents_mask_target = torch.clamp(latents_mask_target, max=0.01)
    else:
        latents_mask_target = latents_mask_source.clone()

    if edit_type == "replace":
        if args.inpaint_ms_minus_mt:
            mask_target = latents_mask_target.unsqueeze(0) == 0.01
            mask_source_minus_target = (latents_mask_source.unsqueeze(0) == 0.01) & (
                ~(mask_target))

            # #Visualize
            # torch.save(mask_source_minus_target, os.path.join(args.output_dir, f"mask_source_minus_target.pt"))

            # if source object has a small relationship with target object
            mask = mask_source_minus_target.view(1, args.height // 16,
                        args.width // 16).unsqueeze(1)

            mask = mask.float()
            kernel = torch.ones(1, 1, 3, 3, device=mask.device)
            for _ in range(2):
                dilated = torch.nn.functional.conv2d(mask.float(), kernel, padding=1)
                mask = (dilated > 0)
            mask_source_minus_target = mask.squeeze(1).view(seq_len)


            x[:, mask_begin[0]] = torch.where(mask_source_minus_target, mask_token_id, x[:, mask_begin[0]])


            vq_mask = x == mask_token_id
            unknown_cnt = vq_mask.sum(dim=1, keepdim=True)
            vq_len = unknown_cnt

            # refine the area in the Mask Target
            for step in range(timesteps):
                if unknown_cnt.item() == 0:
                    break

                # Calculate number of tokens to keep (continue masking) this round
                if step < timesteps - 1:
                    frac = noise_schedule(torch.tensor([(step + 1) / timesteps], device=device))
                    keep_n = (vq_len.float() * frac).floor().clamp_min(1).long()
                else:
                    keep_n = torch.zeros_like(unknown_cnt)

                # Forward pass (with/without CFG)
                if cfg_scale > 0:
                    uncond = torch.cat((uncon_ids.to(x.device), x[:, code_start - 2:]), axis=1)
                    uncond_vq_mask = torch.cat(
                        (torch.zeros((1, uncon_ids.size()[1]), dtype=torch.bool).to(x.device), vq_mask[:, code_start - 2:]),
                        axis=1)
                    cond_logits_raw = model(x, infer=True).logits
                    cond_logits = cond_logits_raw[:, vq_mask[0], vocab_offset: vocab_offset + codebook_size]
                    uncond_logits_raw = model(uncond, infer=True).logits
                    uncond_logits = uncond_logits_raw[:, uncond_vq_mask[0], vocab_offset: vocab_offset + codebook_size]
                    logits = (1 + cfg_scale) * cond_logits - cfg_scale * uncond_logits
                else:
                    logits_raw = model(x, infer=True).logits
                    logits = logits_raw[:, vq_mask[0], vocab_offset: vocab_offset + codebook_size]

                sampled, y0_t_logit = gumbel_max_sample(logits, temperature, generator=generator)
                sampled_full = sampled + vocab_offset
                probs = torch.softmax(logits, dim=-1)
                conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

                flat_idx = vq_mask.nonzero(as_tuple=False)[:, 1]
                x.view(-1)[flat_idx] = sampled_full.view(-1)

                conf_map = torch.full_like(x, -math.inf, dtype=probs.dtype)
                conf_map.view(-1)[flat_idx] = conf.view(-1)

                mask_sel = mask_by_random_topk(keep_n.squeeze(1), conf, temperature=temperature, generator=generator)
                x.view(-1)[flat_idx[mask_sel.view(-1)]] = mask_token_id
                vq_mask = x == mask_token_id
                unknown_cnt = vq_mask.sum(dim=1, keepdim=True)

        # Remove newline tokens
        vq_ids = x[0, code_start:-2]
        vq_ids = vq_ids[vq_ids != newline_id].view(1, seq_len)

    elif edit_type == "add":
        mask_target = (latents_mask_target == 0.01).unsqueeze(0)
        mask = mask_target.view(1, args.height // 16,
                    args.width // 16).float().unsqueeze(1)
        kernel = torch.ones(1, 1, 3, 3, device=mask.device)
        for _ in range(1):
            dilated = torch.nn.functional.conv2d(mask.float(), kernel, padding=1)
            mask = (dilated > 0)
        mask_target = mask.squeeze(1).view(1, seq_len)
        mask_source_minus_target = (latents_mask_source.unsqueeze(0) == 0.01) & (
            ~(mask_target))

        raw_image_ids = raw_image_ids[0, 2:-2]
        raw_image_ids = raw_image_ids[raw_image_ids != newline_id].view(1, seq_len)
        x[:, mask_begin[0]] = torch.where(mask_source_minus_target, raw_image_ids, x[:, mask_begin[0]])

        # # Remove newline tokens
        vq_ids = x[0, code_start:-2]
        vq_ids = vq_ids[vq_ids != newline_id].view(1, seq_len)
    elif edit_type == "remove_ablation":
        vq_ids = x[0, code_start:-2]
        vq_ids = vq_ids[vq_ids != newline_id].view(1, seq_len)
    else:
        raise NotImplementedError

    return vq_ids



@torch.no_grad()
def inpaint_image(
    model,
    prompt: torch.LongTensor,
    args,
    inpaint_con_prompt_ids,

    mask_mode,
    sam2_predictor,
    image,
    target_object_indices,
    con_prompt_ids_2,
    target_object_indices_2,

    mask_inpaint,
    mask_begin: torch.LongTensor,
    *,
    seq_len: int = 1024,
    newline_every: int = 16,
    timesteps: int = 18,
    mask_token_id: int = 126336,
    newline_id: int = 126084,
    temperature: float = 1.0,
    cfg_scale: float = 0.0,
    uncon_ids: torch.LongTensor=None,
    code_start: Optional[int] = None,
    uncon_code_start: Optional[int] = None,
    codebook_size: int = 8192,
    noise_schedule: Callable[[torch.Tensor], torch.Tensor] = cosine_schedule,
    text_vocab_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.LongTensor:

    device = next(model.parameters()).device
    prompt = prompt.to(device)
    B, P = prompt.shape
    assert B == 1, "batch>1 not supported – wrap in loop if needed"

    x = prompt

    # Infer text vocabulary size
    if text_vocab_size is None:
        vocab_total = model(torch.zeros(1, 1, dtype=torch.long, device=device), infer=True).logits.size(-1)
        text_vocab_size = vocab_total - codebook_size
    vocab_offset = text_vocab_size

    if mask_mode == "attention_sam2":
        att_score_list = AttentionScore()
        att_score_list.modify_state(True)
        if target_object_indices == []:
            target_object_indices_used = target_object_indices_2
            input_x = torch.cat((con_prompt_ids_2, x[:, code_start - 2:]), axis=1)
            mask_2 = torch.zeros(
                (mask_begin.shape[0], con_prompt_ids_2.shape[1]),
                dtype=torch.bool,
                device=mask_begin.device
            )
            mask_begin_used = torch.cat((mask_2, mask_begin[:, code_start - 2:]), axis=1)
        else:
            input_x = x.clone()
            target_object_indices_used = target_object_indices
            mask_begin_used = mask_begin.clone()

        _ = model(input_x, infer=True,
                  use_attention_control=True,
                  target_object_indices=target_object_indices_used,
                  image_mask=mask_begin_used.squeeze(0),
                  att_score_list=att_score_list)
        avg_attn = att_score_list.get_avg_att()
        avg_attn = avg_attn - avg_attn.min()
        threshold = avg_attn.max() * 0.8
        subject_mask, point_coords = attention_to_points_sam_predict(avg_attn,
                                                                     (avg_attn > threshold).to(avg_attn.dtype),
                                                                     sam2_predictor, image, device)

        latents_mask = torch.nn.functional.interpolate(
            subject_mask.view(1, 1, subject_mask.shape[-2], subject_mask.shape[-1]),
            size=(subject_mask.shape[-2] // 16, subject_mask.shape[-1] // 16),
            mode='bilinear'
        ).view(subject_mask.shape[-2] // 16, subject_mask.shape[-1] // 16).to(device)
        latents_mask = torch.clamp(latents_mask, max=0.01)
        mask_inpaint = (latents_mask == 0.01).unsqueeze(0)

    mask = mask_inpaint.view(1, args.height // 16,
                        args.width // 16).float().unsqueeze(1)
    kernel = torch.ones(1, 1, 3, 3, device=mask.device)
    for _ in range(2):
        dilated = torch.nn.functional.conv2d(mask.float(), kernel, padding=1)
        mask = (dilated > 0)
    mask_inpaint = mask.squeeze(1).view(seq_len)

    x[:, mask_begin[0]] = torch.where(mask_inpaint, mask_token_id, x[:, mask_begin[0]])

    x = torch.cat((inpaint_con_prompt_ids, x[:, code_start - 2:]), axis=1)
    inpaint_code_start = inpaint_con_prompt_ids.shape[1] + 2
    del code_start
    vq_mask = x == mask_token_id
    unknown_cnt = vq_mask.sum(dim=1, keepdim=True)
    vq_len = unknown_cnt

    #refine the area in the Mask Target
    for step in range(timesteps):
        if unknown_cnt.item() == 0:
            break

        # Calculate number of tokens to keep (continue masking) this round
        if step < timesteps - 1:
            frac = noise_schedule(torch.tensor([(step + 1) / timesteps], device=device))
            keep_n = (vq_len.float() * frac).floor().clamp_min(1).long()
        else:
            keep_n = torch.zeros_like(unknown_cnt)

        # Forward pass (with/without CFG)
        if cfg_scale > 0:
            uncond = torch.cat((uncon_ids.to(x.device), x[:, inpaint_code_start-2:]), axis=1)
            uncond_vq_mask = torch.cat((torch.zeros((1, uncon_ids.size()[1]), dtype=torch.bool).to(x.device), vq_mask[:, inpaint_code_start-2:]), axis=1)
            cond_logits_raw = model(x, infer=True).logits
            cond_logits = cond_logits_raw[:, vq_mask[0], vocab_offset : vocab_offset + codebook_size]
            uncond_logits_raw = model(uncond, infer=True).logits
            uncond_logits = uncond_logits_raw[:, uncond_vq_mask[0], vocab_offset : vocab_offset + codebook_size]
            logits = (1 + cfg_scale) * cond_logits - cfg_scale * uncond_logits
        else:
            logits_raw = model(x, infer=True).logits
            logits = logits_raw[:, vq_mask[0], vocab_offset : vocab_offset + codebook_size]

        sampled, y0_t_logit = gumbel_max_sample(logits, temperature, generator=generator)
        sampled_full = sampled + vocab_offset
        probs = torch.softmax(logits, dim=-1)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        flat_idx = vq_mask.nonzero(as_tuple=False)[:, 1]
        x.view(-1)[flat_idx] = sampled_full.view(-1)

        conf_map = torch.full_like(x, -math.inf, dtype=probs.dtype)
        conf_map.view(-1)[flat_idx] = conf.view(-1)

        mask_sel = mask_by_random_topk(keep_n.squeeze(1), conf, temperature=temperature, generator=generator)
        x.view(-1)[flat_idx[mask_sel.view(-1)]] = mask_token_id
        vq_mask = x == mask_token_id
        unknown_cnt = vq_mask.sum(dim=1, keepdim=True)

    vq_ids = x[0, inpaint_code_start:-2]
    vq_ids = vq_ids[vq_ids != newline_id].view(1, seq_len)

    return vq_ids, x


@torch.no_grad()
def generate_image(
    model,
    prompt: torch.LongTensor,
    *,
    seq_len: int = 1024,
    newline_every: int = 16,
    timesteps: int = 18,
    mask_token_id: int = 126336,
    newline_id: int = 126084,
    temperature: float = 1.0,
    cfg_scale: float = 0.0,
    uncon_ids: torch.LongTensor,
    code_start: Optional[int] = None,
    uncon_code_start: Optional[int] = None,
    codebook_size: int = 8192,
    noise_schedule: Callable[[torch.Tensor], torch.Tensor] = cosine_schedule,
    text_vocab_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    output_dir: str = '',
    if_get_mask_and_zt = False,
) -> torch.LongTensor:
    """
    MaskGit parallel decoding to generate VQ tokens
    
    Args:
        model: Model
        prompt: Prompt tensor
        seq_len: Sequence length
        newline_every: Newline interval per row
        timesteps: Number of timesteps
        mask_token_id: Mask token id
        newline_id: Newline token id
        temperature: Temperature
        cfg_scale: CFG scale
        uncon_ids: Unconditional input
        code_start: Image token satrt index
        codebook_size: Codebook size
        noise_schedule: Noise schedule function
        text_vocab_size: Text vocabulary size
        generator: Random number generator
    
    Returns:
        Final VQ codes (1, seq_len)
    """
    device = next(model.parameters()).device
    prompt = prompt.to(device)
    B, P = prompt.shape
    assert B == 1, "batch>1 not supported – wrap in loop if needed"

    x = prompt
    
    vq_mask = x == mask_token_id
    unknown_cnt = vq_mask.sum(dim=1, keepdim=True)
    vq_len = unknown_cnt

    # Infer text vocabulary size
    if text_vocab_size is None:
        vocab_total = model(torch.zeros(1, 1, dtype=torch.long, device=device), infer=True).logits.size(-1)
        text_vocab_size = vocab_total - codebook_size
    vocab_offset = text_vocab_size

    for step in range(timesteps):
        if if_get_mask_and_zt:
            torch.save(vq_mask[:, code_start-2:], os.path.join(output_dir, f"mask_step_{step}.pt"))

        mask_now = vq_mask.clone()

        if unknown_cnt.item() == 0:
            break

        # Calculate number of tokens to keep (continue masking) this round
        if step < timesteps - 1:
            frac = noise_schedule(torch.tensor([(step + 1) / timesteps], device=device))
            keep_n = (vq_len.float() * frac).floor().clamp_min(1).long()
        else:
            keep_n = torch.zeros_like(unknown_cnt)

        # Forward pass (with/without CFG)
        if cfg_scale > 0:
            uncond = torch.cat((uncon_ids.to(x.device), x[:, code_start-2:]), axis=1)
            uncond_vq_mask = torch.cat((torch.zeros((1, uncon_ids.size()[1]), dtype=torch.bool).to(x.device), vq_mask[:, code_start-2:]), axis=1)
            cond_logits_raw = model(x, infer=True).logits
            cond_logits = cond_logits_raw[:, vq_mask[0], vocab_offset : vocab_offset + codebook_size]
            uncond_logits_raw = model(uncond, infer=True).logits
            uncond_logits = uncond_logits_raw[:, uncond_vq_mask[0], vocab_offset : vocab_offset + codebook_size]
            logits = (1 + cfg_scale) * cond_logits - cfg_scale * uncond_logits

            cond_logits_full = cond_logits_raw[:, code_start-2:, vocab_offset: vocab_offset + codebook_size]
            uncond_logits_full = uncond_logits_raw[:, uncon_code_start-2:, vocab_offset: vocab_offset + codebook_size]
            logits_full = (1 + cfg_scale) * cond_logits_full - cfg_scale * uncond_logits_full

        else:
            logits_raw = model(x, infer=True).logits
            logits = logits_raw[:, vq_mask[0], vocab_offset : vocab_offset + codebook_size]
            logits_full = logits_raw[:, code_start-2:, vocab_offset: vocab_offset + codebook_size]

        sampled, y0_t_logit = gumbel_max_sample(logits, temperature, generator=generator)

        if if_get_mask_and_zt:
            y0 = construct_ground_truth_y0(sampled, y0_t_logit)
            zt = y0 - y0_t_logit
            torch.save(zt, os.path.join(output_dir, f"z_step_{step}.pt"))

        sampled_full = sampled + vocab_offset
        probs = torch.softmax(logits, dim=-1)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        flat_idx = vq_mask.nonzero(as_tuple=False)[:, 1]
        x.view(-1)[flat_idx] = sampled_full.view(-1)

        conf_map = torch.full_like(x, -math.inf, dtype=probs.dtype)
        conf_map.view(-1)[flat_idx] = conf.view(-1)

        mask_sel = mask_by_random_topk(keep_n.squeeze(1), conf, temperature=temperature, generator=generator)
        x.view(-1)[flat_idx[mask_sel.view(-1)]] = mask_token_id
        vq_mask = x == mask_token_id
        unknown_cnt = vq_mask.sum(dim=1, keepdim=True)

    # Remove newline tokens
    vq_ids = x[0, code_start:-2]
    vq_ids = vq_ids[vq_ids != newline_id].view(1, seq_len)
    return vq_ids



class AttentionScore:
    def __init__(self):
        self.attention_score = []
        self.avg_att = None
        self.allowed = False

    def get_state(self):
        return self.allowed

    def modify_state(self, is_enbale):
        self.allowed = is_enbale

    def get_avg_att(self):
        att = torch.stack(self.attention_score, dim=0)  # [N, 1, 32, 1, 1024]
        att = att.view(-1, att.size(-1))  # [N*1*32*1, 1024]
        self.avg_att = att.mean(dim=0)
        return self.avg_att

    def reset_att(self):
        self.avg_att = None
        self.attention_score = []
        self.allowed = False

    def add_attention_score(self, att):
        assert self.allowed
        assert self.avg_att == None
        self.attention_score.append(att)

## cite from https://github.com/NVlabs/addit/blob/main/addit_blending_utils.py#L149
def attention_to_points_sam_predict(subject_attention, subject_mask, sam_predictor, image, device):
    H, W = image.size

    # Resize clipseg mask to image size
    subject_attention = F.interpolate(
        subject_attention.view(1, 1, H // 16, W // 16), size=(H, W),
        mode='bilinear').view(H, W)
    subject_mask = F.interpolate(subject_mask.view(1, 1, H // 16, W // 16), size=(H, W),
                                 mode='bilinear').view(H, W)

    # Get mask_bbox
    subject_mask_indices = torch.nonzero(subject_mask)
    top_left = subject_mask_indices.min(dim=0)[0]
    bottom_right = subject_mask_indices.max(dim=0)[0]
    box_width = bottom_right[1] - top_left[1]
    box_height = bottom_right[0] - top_left[0]

    # Define the number of points and minimum distance between points
    n_points = 3
    max_thr = 0.35
    max_attention = torch.max(subject_attention)
    min_distance = max(box_width, box_height) // (n_points + 1)  # Adjust this value to control spread
    # min_distance = max(min_distance, 75)

    # Initialize list to store selected points
    selected_points = []

    # Create a copy of the attention map
    remaining_attention = subject_attention.clone()

    for _ in range(n_points):
        if remaining_attention.max() < max_thr * max_attention:
            break

        # Find the highest attention point
        point = torch.argmax(remaining_attention)
        y, x = torch.unravel_index(point, remaining_attention.shape)
        y, x = y.item(), x.item()

        # Add the point to our list
        selected_points.append((x, y))

        # Zero out the area around the selected point
        y_min = max(0, y - min_distance)
        y_max = min(H, y + min_distance + 1)
        x_min = max(0, x - min_distance)
        x_max = min(W, x + min_distance + 1)
        remaining_attention[y_min:y_max, x_min:x_max] = 0

    # Convert selected points to numpy array
    point_coords = np.array(selected_points)
    point_labels = np.ones(point_coords.shape[0], dtype=int)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_predictor.set_image(image)
        masks, scores, logits = sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )

    subject_mask = torch.tensor(masks[0], device=device)

    return subject_mask, point_coords

@torch.no_grad()
def get_inversion(
        model,
        prompt: torch.LongTensor,
        mask_begin: torch.LongTensor,
        *,
        use_attention_control=False,
        latents_mask=None,

        args=None,
        mask_mode=None,
        sam2_predictor=None,
        image=None,
        target_object_indices=[],
        con_prompt_ids_2=None,
        target_object_indices_2=[],

        seq_len: int = 1024,
        newline_every: int = 16,
        timesteps: int = 18,
        mask_token_id: int = 126336,
        newline_id: int = 126084,
        temperature: float = 1.0,
        cfg_scale: float = 0.0,
        uncon_ids: torch.LongTensor,
        code_start: Optional[int] = None,
        uncon_code_start: Optional[int] = None,
        codebook_size: int = 8192,
        noise_schedule: Callable[[torch.Tensor], torch.Tensor] = sine_schedule,
        text_vocab_size: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        output_dir: str = '',
) -> torch.LongTensor:
    """
    MaskGit parallel decoding to generate VQ tokens

    Args:
        model: Model
        prompt: Prompt tensor
        seq_len: Sequence length
        newline_every: Newline interval per row
        timesteps: Number of timesteps
        mask_token_id: Mask token id
        newline_id: Newline token id
        temperature: Temperature
        cfg_scale: CFG scale
        uncon_ids: Unconditional input
        code_start: Image token satrt index
        codebook_size: Codebook size
        noise_schedule: Noise schedule function
        text_vocab_size: Text vocabulary size
        generator: Random number generator

    Returns:
        Final VQ codes (1, seq_len)
    """
    device = next(model.parameters()).device
    prompt = prompt.to(device)
    B, P = prompt.shape
    assert B == 1, "batch>1 not supported – wrap in loop if needed"

    x = prompt

    vq_mask = mask_begin.clone()

    ## here modified
    # known_cnt = vq_mask.sum(dim=1, keepdim=True)
    # vq_len = known_cnt

    # Infer text vocabulary size
    if text_vocab_size is None:
        vocab_total = model(torch.zeros(1, 1, dtype=torch.long, device=device), infer=True).logits.size(-1)
        text_vocab_size = vocab_total - codebook_size
    vocab_offset = text_vocab_size

    ground_truth_image_part = x[:, code_start - 2:].clone()
    if use_attention_control:
        if mask_mode == "attention_sam2":
            att_score_list = AttentionScore()
            att_score_list.modify_state(True)
            if target_object_indices == []:
                target_object_indices_used = target_object_indices_2
                input_x = torch.cat((con_prompt_ids_2, x[:, code_start - 2:]), axis=1)
                mask_2 = torch.zeros(
                    (mask_begin.shape[0], con_prompt_ids_2.shape[1]),
                    dtype=torch.bool,
                    device=mask_begin.device
                )
                mask_begin_used = torch.cat((mask_2, mask_begin[:, code_start - 2:]), axis=1)
            else:
                input_x = x.clone()
                target_object_indices_used = target_object_indices
                mask_begin_used = mask_begin.clone()

            _ = model(input_x, infer=True,
                    use_attention_control=True,
                    target_object_indices=target_object_indices_used,
                    image_mask=mask_begin_used.squeeze(0),
                    att_score_list=att_score_list)
            avg_attn = att_score_list.get_avg_att()
            avg_attn = avg_attn - avg_attn.min()
            threshold = avg_attn.max() * 0.8

            subject_mask, point_coords = attention_to_points_sam_predict(avg_attn,
                        (avg_attn > threshold).to(avg_attn.dtype), sam2_predictor, image, device)

            latents_mask = torch.nn.functional.interpolate(
                subject_mask.view(1, 1, subject_mask.shape[-2], subject_mask.shape[-1]),
                size=(subject_mask.shape[-2] // 16, subject_mask.shape[-1] // 16),
                mode='bilinear'
            ).view(seq_len).to(device)
            latents_mask = torch.clamp(latents_mask, max=0.01)

    if_less_gap = latents_mask.unsqueeze(0)
    vq_mask[mask_begin] = if_less_gap == 0.01
    mask_begin = vq_mask.clone()

    #temporal
    known_cnt = vq_mask.sum(dim=1, keepdim=True)
    vq_len = known_cnt
    # end of temporal


    for step in range(timesteps+1):
        # Forward pass (with/without CFG)
        if cfg_scale > 0:
            uncond = torch.cat((uncon_ids.to(x.device), x[:, code_start - 2:]), axis=1)
            uncond_vq_mask = torch.cat(
                (torch.zeros((1, uncon_ids.size()[1]), dtype=torch.bool).to(x.device), vq_mask[:, code_start - 2:]),
                axis=1)

            output = model(x, infer=True)
            cond_logits_raw = output.logits
            cond_logits = cond_logits_raw[:, vq_mask[0], vocab_offset: vocab_offset + codebook_size]
            uncond_logits_raw = model(uncond, infer=True).logits
            uncond_logits = uncond_logits_raw[:, uncond_vq_mask[0], vocab_offset: vocab_offset + codebook_size]
            logits = (1 + cfg_scale) * cond_logits - cfg_scale * uncond_logits

            cond_logits_full = cond_logits_raw[:, code_start - 2:, vocab_offset: vocab_offset + codebook_size]
            uncond_logits_full = uncond_logits_raw[:, uncon_code_start - 2:, vocab_offset: vocab_offset + codebook_size]
            logits_full = (1 + cfg_scale) * cond_logits_full - cfg_scale * uncond_logits_full


        else:
            output = model(x, infer=True)
            logits_raw = output.logits
            logits = logits_raw[:, vq_mask[0], vocab_offset: vocab_offset + codebook_size]
            logits_full = logits_raw[:, code_start - 2:, vocab_offset: vocab_offset + codebook_size]

        sampled, _ = gumbel_max_sample(logits, temperature, generator=generator)
        _, y0_t_logit = gumbel_max_sample(logits_full, temperature, generator=generator)

        if step > 0:
            ground_truth_label = ground_truth_image_part[:, mask_save[0]] - vocab_offset
            y0_partial = y0_t_logit[:, mask_save[0]]

            y0 = construct_ground_truth_y0(ground_truth_label, y0_partial)
            zt = y0 - y0_partial
            torch.save(zt, os.path.join(output_dir, f"z_step_{timesteps - step}.pt"))

        if known_cnt.item() == 0 and step < timesteps:
            break

        if step == timesteps:
            break

        # Calculate number of tokens to keep (continue masking) this round
        frac_pre = noise_schedule(torch.tensor([(step) / timesteps], device=device))
        mask_n_pre = (vq_len.float() * frac_pre).floor().clamp_min(1).long()
        frac_now = noise_schedule(torch.tensor([(step + 1) / timesteps], device=device))
        mask_n_now = (vq_len.float() * frac_now).floor().clamp_min(1).long()
        mask_n = mask_n_now - mask_n_pre

        probs = torch.softmax(logits, dim=-1)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        flat_idx = vq_mask.nonzero(as_tuple=False)[:, 1]
        mask_sel = mask_by_random_topk_inversion(mask_n.squeeze(1), conf, temperature=temperature, generator=generator)

        if step < timesteps - 1:
            x.view(-1)[flat_idx[mask_sel.view(-1)]] = mask_token_id
        else:
            x[mask_begin] = mask_token_id
        vq_mask = (vq_mask & (x != mask_token_id))
        known_cnt = vq_mask.sum(dim=1, keepdim=True)

        mask_save = mask_begin[:, code_start - 2:] & (~vq_mask[:, code_start - 2:])
        torch.save(mask_save
                   , os.path.join(output_dir, f"mask_step_{timesteps - 1 - step}.pt"))

    return latents_mask


def construct_ground_truth_y0(ground_truth_label, y0_partial):
    r_value = y0_partial[0, torch.arange(y0_partial.size(1)), ground_truth_label[0]]
    gumbel_dist_r = torch.distributions.gumbel.Gumbel(r_value, torch.ones_like(r_value))
    r_value_gumbel = gumbel_dist_r.sample()

    y0_out = torch.zeros(y0_partial.size(), dtype=y0_partial.dtype, device=y0_partial.device)
    y0_out[0, torch.arange(y0_partial.size(1)), ground_truth_label[0]] = r_value_gumbel

    seq_len = y0_partial.size(1)
    vocab_size = y0_partial.size(2)
    all_indices = torch.arange(vocab_size, device=y0_partial.device).unsqueeze(0).expand(seq_len, vocab_size)
    mask = all_indices != ground_truth_label[0].unsqueeze(1)

    r_value_gumbel_expanded = r_value_gumbel.unsqueeze(1).expand(-1, vocab_size - 1)
    tmp_location = r_value_gumbel_expanded - 0.1 * r_value_gumbel_expanded.abs()
    loc = (0.2 * r_value_gumbel_expanded).abs()

    scale = torch.ones_like(tmp_location)
    gumbel_dist_y0_exclude_gt = torch.distributions.gumbel.Gumbel(loc, scale)

    y0_exclude_gt = tmp_location - gumbel_dist_y0_exclude_gt.sample().abs()
    y0_out[:, mask] = y0_exclude_gt.flatten().to(y0_out.dtype)
    return y0_out




@torch.no_grad()
def get_inversion_dice(
        model,
        prompt: torch.LongTensor,
        mask_begin: torch.LongTensor,
        *,
        seq_len: int = 1024,
        newline_every: int = 16,
        timesteps: int = 18,
        mask_token_id: int = 126336,
        newline_id: int = 126084,
        temperature: float = 1.0,
        cfg_scale: float = 0.0,
        uncon_ids: torch.LongTensor,
        code_start: Optional[int] = None,
        uncon_code_start: Optional[int] = None,
        codebook_size: int = 8192,
        noise_schedule: Callable[[torch.Tensor], torch.Tensor] = sine_schedule,
        text_vocab_size: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        output_dir: str = '',
):
    device = next(model.parameters()).device
    prompt = prompt.to(device)
    B, P = prompt.shape
    assert B == 1, "batch>1 not supported – wrap in loop if needed"
    x = prompt
    vq_mask = mask_begin.clone()

    known_cnt = vq_mask.sum(dim=1, keepdim=True)
    vq_len = known_cnt

    # Infer text vocabulary size
    if text_vocab_size is None:
        vocab_total = model(torch.zeros(1, 1, dtype=torch.long, device=device), infer=True).logits.size(-1)
        text_vocab_size = vocab_total - codebook_size
    vocab_offset = text_vocab_size

    mask_begin = vq_mask.clone()

    for step in range(timesteps+1):
        # Forward pass (with/without CFG)
        if cfg_scale > 0:
            uncond = torch.cat((uncon_ids.to(x.device), x[:, code_start - 2:]), axis=1)
            uncond_vq_mask = torch.cat(
                (torch.zeros((1, uncon_ids.size()[1]), dtype=torch.bool).to(x.device), vq_mask[:, code_start - 2:]),
                axis=1)

            output = model(x, infer=True)
            cond_logits_raw = output.logits

            cond_logits = cond_logits_raw[:, vq_mask[0], vocab_offset: vocab_offset + codebook_size]
            uncond_logits_raw = model(uncond, infer=True).logits
            uncond_logits = uncond_logits_raw[:, uncond_vq_mask[0], vocab_offset: vocab_offset + codebook_size]
            logits = (1 + cfg_scale) * cond_logits - cfg_scale * uncond_logits

            cond_logits_full = cond_logits_raw[:, code_start - 2:, vocab_offset: vocab_offset + codebook_size]
            uncond_logits_full = uncond_logits_raw[:, uncon_code_start - 2:, vocab_offset: vocab_offset + codebook_size]
            logits_full = (1 + cfg_scale) * cond_logits_full - cfg_scale * uncond_logits_full


        else:
            output = model(x, infer=True)
            logits_raw = output.logits
            logits = logits_raw[:, vq_mask[0], vocab_offset: vocab_offset + codebook_size]
            logits_full = logits_raw[:, code_start - 2:, vocab_offset: vocab_offset + codebook_size]

        sampled, _ = gumbel_max_sample(logits, temperature, generator=generator)
        _, y0_t_logit = gumbel_max_sample(logits_full, temperature, generator=generator)

        if step > 0:
            y0_partial = y0_t_logit[:, mask_save[0]]
            y0 = torch.load(os.path.join(output_dir, "y0_logit.pt"))[:, mask_save[0]]
            zt = y0 - y0_partial
            torch.save(zt, os.path.join(output_dir, f"z_step_{timesteps - step}.pt"))

        if known_cnt.item() == 0 and step < timesteps:
            break

        if step == timesteps:
            break

        # Calculate number of tokens to keep (continue masking) this round
        frac_pre = noise_schedule(torch.tensor([(step) / timesteps], device=device))
        mask_n_pre = (vq_len.float() * frac_pre).floor().clamp_min(1).long()
        frac_now = noise_schedule(torch.tensor([(step + 1) / timesteps], device=device))
        mask_n_now = (vq_len.float() * frac_now).floor().clamp_min(1).long()
        mask_n = mask_n_now - mask_n_pre

        probs = torch.softmax(logits, dim=-1)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        flat_idx = vq_mask.nonzero(as_tuple=False)[:, 1]

        conf_map = torch.full_like(x, -math.inf, dtype=probs.dtype)
        conf_map.view(-1)[flat_idx] = conf.view(-1)

        mask_sel = mask_by_random_topk_inversion(mask_n.squeeze(1), conf, temperature=temperature, generator=generator)

        if step < timesteps - 1:
            x.view(-1)[flat_idx[mask_sel.view(-1)]] = mask_token_id
        else:
            x[mask_begin] = mask_token_id
        vq_mask = (vq_mask & (x != mask_token_id))
        known_cnt = vq_mask.sum(dim=1, keepdim=True)

        mask_save = mask_begin[:, code_start - 2:] & (~vq_mask[:, code_start - 2:])
        torch.save(mask_save
                   , os.path.join(output_dir, f"mask_step_{timesteps - 1 - step}.pt"))
