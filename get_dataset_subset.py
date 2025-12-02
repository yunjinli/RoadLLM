import os
import sys
import json
from glob import glob
import random
import shutil
from pathlib import Path

random.seed(42)
number_samples_per_dataset = 100
dataset_base_config_path = './configs/dataset_paths.json'
with open(dataset_base_config_path) as f:
    dataset_base_config = json.load(f)
    print(dataset_base_config)

output_dir = "./dataset_subset"
json_base = "/home/phd_li/dataset/roadllm"


# os.makedirs(output_dir, exist_ok=True)

for k, v in dataset_base_config.items():
    # os.makedirs(os.path.join(output_dir, k), exist_ok=True)
    # for _ in range(number_samples_per_dataset):
    caption_dataset_base_path = os.path.join(json_base, k)
    json_cap_list = sorted(glob(os.path.join(caption_dataset_base_path, '*.json')))
    json_cap_list_sampled = random.sample(json_cap_list, number_samples_per_dataset)
    for json_cap_path in json_cap_list_sampled:
        ds_dir = Path(os.path.join(output_dir, k))
        ds_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(json_cap_path, ds_dir)
        with open(json_cap_path) as f:
            cap = json.load(f)
        
        ds_dir = Path(os.path.join(output_dir, 'images', k, cap['image'])).parent
        ds_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(os.path.join(v, cap['image']), ds_dir)
        