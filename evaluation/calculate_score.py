import re
import os
import numpy as np

def compute_score(text):
    prompt_compliance = list(map(int, re.findall(r"Prompt Compliance:\s*(\d+)", text)))
    visual_naturalness = list(map(int, re.findall(r"Visual Naturalness:\s*(\d+)", text)))

    assert len(prompt_compliance) == 2
    assert len(visual_naturalness) == 2
    avg_prompt = sum(prompt_compliance) / len(prompt_compliance)
    avg_visual = sum(visual_naturalness) / len(visual_naturalness)
    return avg_prompt, avg_visual

all_avg_prompt = []
all_avg_visual = []

model = "" # your evaluated model name, e.g., "gpt"
base_target_img_folder = f"evaluation_result_{model}"
all_imgs = os.listdir(base_target_img_folder)
img_indices = sorted(
    int(os.path.splitext(name)[0])
    for name in all_imgs
    if name.lower().endswith((".jpg", ".png", ".jpeg")) and os.path.splitext(name)[0].isdigit()
)
print(len(img_indices))
for i in img_indices:
    txt_path = f"evaluation_score/{model}/{i}.txt"
    try:
        with open(txt_path, "r", encoding="utf-8") as file:
            text = file.read()
        tmp_prompt, tmp_visual = compute_score(text)
        # tmp_visual = min(tmp_visual, tmp_prompt)
        all_avg_prompt.append(tmp_prompt)
        all_avg_visual.append(tmp_visual)
    except:
        print(i)


print(f"Prompt Compliance: {np.mean(all_avg_prompt)}")
print(f"Visual Naturalness: {np.mean(all_avg_visual)}")

