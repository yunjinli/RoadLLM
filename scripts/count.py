from glob import glob
import os
base_path = '/home/phd_li/dataset/roadllm'
paths = glob(os.path.join(base_path, '**', '*.json'), recursive=True)

print(len(paths))