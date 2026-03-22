import os
import sys
import json
import cv2
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image
import torch

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage.metrics import structural_similarity as ssim
import numbers

def align_images(original, edited):
    gray1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(gray1, None)
    k2, d2 = sift.detectAndCompute(gray2, None)
    if d1 is None or d2 is None:
        return None
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(d1, d2, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if len(good) < 4:
        return None
    src_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.LMEDS)
    if matrix is None:
        return None
    h, w = original.shape[:2]
    return cv2.warpAffine(edited, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def compute_aligned_ssim_outside_mask(original, aligned, mask_array):
    def normalize_image(x):
        return np.array(x, dtype=np.float32) / 255.0

    if aligned is None or mask_array is None:
        return None

    if aligned.shape != original.shape:
        aligned = cv2.resize(aligned, (original.shape[1], original.shape[0]))

    mask = np.array(mask_array, dtype=np.uint8)
    if mask.ndim > 2:
        mask = mask[:, :, 0]

    if mask.shape != original.shape[:2]:
        mask = cv2.resize(
            mask,
            (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    rest_mask = (mask == 0).astype(np.float32)

    if rest_mask.sum() == 0:
        return None

    if original.ndim == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        aligned_gray = aligned

    original_masked = normalize_image(original_gray) * rest_mask
    aligned_masked  = normalize_image(aligned_gray) * rest_mask

    ssim_value, ssim_map = ssim(
        original_masked,
        aligned_masked,
        data_range=1.0,
        full=True
    )


    return ssim_value


sam3_model = build_sam3_image_model(
    checkpoint_path="your_sam3_model_dir/sam3.pt" # Please replace with your actual SAM3 model path
)
sam3_processor = Sam3Processor(sam3_model)

device = "cuda"
grounding_dino_model_id = "IDEA-Research/grounding-dino-base"
grounding_processor = AutoProcessor.from_pretrained(grounding_dino_model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_dino_model_id).to(device)
sam2_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

def get_ssim(original_path, edited_path, entity_list, edit_type_list):
    original = cv2.imread(original_path)
    edited = cv2.imread(edited_path)
    aligned = align_images(original, edited)
    if aligned is None:
        aligned = cv2.resize(edited, (original.shape[1], original.shape[0]))

    all_masks = []
    for i in range(len(entity_list)):
        ent_list = entity_list[i]
        if edit_type_list[i] != "add":
            img = Image.open(original_path)
        else:
            aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(aligned_rgb)

        for ent in ent_list:
            inference_state = sam3_processor.set_image(img)
            output = sam3_processor.set_text_prompt(state=inference_state, prompt=ent)
            masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
            ent_mask = masks.any(dim=0, keepdim=True)
            ent_mask = ent_mask.view(ent_mask.shape[-2], ent_mask.shape[-1])

            if not ent_mask.any():
                inputs = grounding_processor(images=img, text=ent, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = grounding_model(**inputs)
                results = grounding_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=0.4,
                    text_threshold=0.3,
                    target_sizes=[img.size[::-1]]
                )
                input_boxes = results[0]["boxes"].cpu().numpy()

                if input_boxes.shape[0] != 0:
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        sam2_predictor.set_image(img)
                        masks, scores, logits = sam2_predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_boxes,
                            multimask_output=False,
                        )
                    ent_mask = torch.tensor(masks[0], device=device)
                    ent_mask = ent_mask.view(ent_mask.shape[-2], ent_mask.shape[-1])
            all_masks.append(ent_mask)

    subject_mask = torch.stack(all_masks, dim=0).any(dim=0)
    mask_np = subject_mask.view(subject_mask.shape[-2], subject_mask.shape[-1]).cpu().numpy().astype(np.uint8)
    res = compute_aligned_ssim_outside_mask(original, aligned, mask_np)
    return res


with open("../GIDE-Bench/edit_instructions.json", "r", encoding="utf-8") as f:
    edit_ins = json.load(f)

with open("../GIDE-Bench/input_all.json", "r", encoding="utf-8") as f:
    input_all = json.load(f)

bench_base_folder = "../GIDE-Bench"
source_img_folder = os.path.join(bench_base_folder, "img")

models = [""] # Please replace with the actual model names you want to evaluate, e.g., ["model1", "model2", "model3"]

saved_dict = {}
save_path = "../GIDE-Bench/ssim_results.json"
if os.path.exists(save_path):
    with open(save_path, "r", encoding="utf-8") as f:
        saved_dict = json.load(f)

for model in models:
    generated_img_folder = os.path.join(f"evaluation_result_{model}")
    all_imgs = os.listdir(generated_img_folder)
    img_indices = sorted(
        int(os.path.splitext(name)[0])
        for name in all_imgs
        if name.endswith((".png", ".jpg"))
        and os.path.splitext(name)[0].isdigit()
    )

    all_ssim = {}
    tmp_saved_base_folder = "../GIDE-Bench/tmp_ssim_folder"
    os.makedirs(tmp_saved_base_folder, exist_ok=True)
    tmp_ssim_path = os.path.join(tmp_saved_base_folder, f"tmp_{model}_all_ssim.json")
    if os.path.exists(tmp_ssim_path):
        with open(tmp_ssim_path, "r", encoding="utf-8") as f:
            all_ssim = json.load(f)

    for index in img_indices:
        if str(index) in all_ssim:
            continue

        original_path = os.path.join(source_img_folder, f"{index}.jpg")
        edited_path = os.path.join(generated_img_folder, f"{index}.jpg")
        if not os.path.exists(edited_path):
            edited_path = os.path.join(generated_img_folder, f"{index}.png")

        edit_type_list = edit_ins[str(index)]["edit_type"].split("_")

        subject_token_all = input_all[str(index)]["subject_token"]
        entity_list = []
        for i in range(len(edit_type_list)):
            edit_type = edit_type_list[i]
            if edit_type == "add":
                entity_list.append([subject_token_all[f"instruction_{i + 1}"]["target_subject_token"]])
            elif edit_type == "replace":
                entity_list.append([subject_token_all[f"instruction_{i + 1}"]["target_subject_token"],
                            subject_token_all[f"instruction_{i + 1}"]["source_subject_token"]])
            elif edit_type == "remove":
                entity_list.append([subject_token_all[f"instruction_{i + 1}"]["source_subject_token"]])
            else:
                raise ValueError

        res = get_ssim(original_path, edited_path, entity_list, edit_type_list)
        all_ssim[str(index)] = res
        with open(tmp_ssim_path, "w") as f:
            json.dump(all_ssim, f, indent=2)

    valid_values = [
        v for v in all_ssim.values()
        if isinstance(v, numbers.Number)
    ]
    saved_dict[model] = sum(valid_values) / len(valid_values)
    with open(save_path, "w") as f:
        json.dump(saved_dict, f, indent=2)