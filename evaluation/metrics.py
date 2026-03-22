import os
import requests
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

api_key = "" # your OpenAI API key, e.g., "sk-xxxx"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_scores(source_img_path, target_image_path, edit_types, edit_ins):
    base64_source_img = encode_image(source_img_path)
    base64_target_img = encode_image(target_image_path)

    input_prompt = f"""
       You are a data rater specializing in grading multi-step image edits.
       You will be given:
       - An original image
       - An edited image
       - One editing instruction that contains TWO sub-instructions
         (each being one of: add, replace, remove)

       Your tasks:

       1. Parse the editing instruction into two sub-instructions.
       2. For each sub-instruction:
          - Identify its edit type: add / replace / remove.
          - Evaluate ONLY the visual change related to this sub-instruction.
          - Use the correct scoring rubric (provided below).
       3. Output TWO SEPARATE SCORES (one for each sub-instruction):
          Each score MUST contain:
            • Brief reasoning (≤ 20 words)
            • Prompt Compliance (1–5)
            • Visual Naturalness (1–5)
       4. Output a final average score = mean of the TWO Prompt Compliance scores.

       ------------------------------------------------------------
       SCORING RUBRICS
       ------------------------------------------------------------

       ========================
       REPLACE RUBRIC
       ========================
       Prompt Compliance
       1  Target not replaced, or an unrelated object edited.
       2  Only part of the target replaced, or wrong class/description used.
       3  Target largely replaced but other objects altered, remnants visible, or count/position clearly wrong.
       4  Correct object fully replaced; only minor attribute errors (colour, size, etc.).
       5  Perfect replacement: all and only the specified objects removed; new objects’ class, number, position, scale, pose and detail exactly match the prompt.

       Visual Naturalness
       1  Image heavily broken or new object deformed / extremely blurred.
       2  Obvious seams, smears, or strong mismatch in resolution or colour; background not restored.
       3  Basic style similar, but lighting or palette clashes; fuzzy edges or noise are noticeable.
       4  Style almost uniform; tiny edge artefacts visible only on close inspection; casual viewers see no edit.
       5  Completely seamless; new objects blend fully with the scene, edit area undetectable.

       ========================
       ADD RUBRIC
       ========================
       Prompt Compliance
       1  Nothing added or the added content is corrupt.
       2  Added object is a wrong class or unrelated to the prompt.
       3  Correct class, but key attributes (position, colour, size, count, etc.) are wrong.
       4  Main attributes correct; only minor details off or 1-2 small features missing.
       5  Every stated attribute correct and scene logic reasonable; only microscopic flaws.

       Visual Naturalness
       1  Image badly broken or full of artefacts.
       2  Obvious paste marks; style, resolution, or palette strongly mismatch.
       3  General style similar, but lighting or colours clearly clash; noticeable disharmony.
       4  Style almost uniform; small edge issues visible only when zoomed.
       5  Perfect blend; no visible difference between added object and original image.

       ========================
       REMOVE RUBRIC
       ========================
       Prompt Compliance
       1  Nothing removed, or an unrelated object edited.
       2  Target only partly removed, or a different instance/class deleted, or another object appears in the gap.
       3  Target mostly removed but extra objects also deleted, or fragments of the target remain.
       4  Only the specified objects removed, but a few tiny background items deleted by mistake, or the count is wrong.
       5  Perfect: all and only the requested objects removed; every other element untouched.

       Visual Naturalness
       1  Image badly broken (large holes, strong artefacts).
       2  Clear erase marks; colour/resolution mismatch; background not restored.
       3  General look acceptable yet lighting/colour/style still clash; blur or noise visible.
       4  Style consistent; minor edge issues visible only when zoomed.
       5  Seamless: removal is virtually impossible to spot.

       ------------------------------------------------------------
       MANDATORY OUTPUT FORMAT
       ------------------------------------------------------------
       Sub-instruction 1:
       - Type: {edit_types[0]}
       - Brief reasoning:
       - Prompt Compliance:
       - Visual Naturalness:

       Sub-instruction 2:
       - Type: {edit_types[1]}
       - Brief reasoning:
       - Prompt Compliance:
       - Visual Naturalness:

       Final Average Score (based on compliance only):

       Editing instruction: {edit_ins}

       Below are the images before and after editing:
       """

    payload = {
        "model": 'gpt-5.1',
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": input_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_source_img}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_target_img}"
                        }
                    }
                ]}
        ],
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_data = response.json()["choices"][0]["message"]["content"]
    return response_data

def process_single_index(
    i,
    model,
    edit_ins_all,
    base_source_img_folder,
    base_target_img_folder,
    saved_path_base,
    max_retry=3,
    if_indirect=False
):
    outfile_path = os.path.join(saved_path_base, f"{i}.txt")
    if os.path.exists(outfile_path):
        return i, "skipped"

    if if_indirect:
        item = edit_ins_all[str(i)]
        edit_types = item[0].split("_", 1)
        tmp_edit_ins = item[1] + " " + item[2]
    else:
        item = edit_ins_all[str(i)]
        edit_types = item["edit_type"].split("_", 1)
        tmp_edit_ins = item["evaluation"]

    assert len(edit_types) == 2
    source_img_path = os.path.join(base_source_img_folder, f"{i}.jpg")
    target_img_path = os.path.join(base_target_img_folder, f"{i}.jpg")

    response_data = None
    for retry in range(max_retry):
        try:
            response_data = get_scores(
                source_img_path,
                target_img_path,
                edit_types,
                tmp_edit_ins
            )
            break
        except Exception as e:
            print(f"[{model}] idx {i} failed (retry {retry+1}/{max_retry}): {e}")

    if response_data is None:
        return i, "failed"

    with open(outfile_path, "w", encoding="utf-8") as f:
        f.write(response_data)

    return i, "done"

models = [""] # your evaluated model names, e.g., ["gpt", "MMaDA", "ours"]
for model in models:
    ins_path = "../GIDE-Bench/edit_instructions.json"
    with open(ins_path, "r", encoding="utf-8") as f:
        edit_ins_all = json.load(f)
    base_source_img_folder = "../GIDE-Bench/img"
    base_target_img_folder = f"evaluation_result_{model}"
    saved_path_base = f"evaluation_score/{model}"
    os.makedirs(saved_path_base, exist_ok=True)

    img_indices = sorted(
        int(os.path.splitext(name)[0])
        for name in os.listdir(base_target_img_folder)
        if name.endswith(".jpg") and os.path.splitext(name)[0].isdigit()
    )

    if_indirect = False
    num_threads=32
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(
                process_single_index,
                i,
                model,
                edit_ins_all,
                base_source_img_folder,
                base_target_img_folder,
                saved_path_base,
                if_indirect=if_indirect,
            ): i
            for i in img_indices
        }

        for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Scoring {model}"
        ):
            i = futures[future]
            try:
                idx, status = future.result()
                if status != "done":
                    print(f"[{model}] idx {idx}: {status}")
            except Exception as e:
                print(f"[{model}] idx {i} crashed: {e}")