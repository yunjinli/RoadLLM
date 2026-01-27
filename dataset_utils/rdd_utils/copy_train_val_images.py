import sys
import os
from tqdm import tqdm
import shutil


base_dir, mode, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]



with open(os.path.join(base_dir, f'{mode}_annotation_path.txt'), 'r') as f:
    annotation_path = f.read().split()

print(f"Copying {len(annotation_path)} {mode} samples...")
os.makedirs(os.path.join(output_dir, f'{mode}2017'), exist_ok=True)

for path in tqdm(annotation_path):
    img_path = path.replace('annotations/xmls', 'images').replace('.xml', '.jpg')
    dst_img_path = os.path.join(output_dir, f'{mode}2017', img_path.split('/')[-1])
    # print(dst_img_path)
    shutil.copyfile(img_path, dst_img_path)


