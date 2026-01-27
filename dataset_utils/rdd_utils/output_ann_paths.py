import sys
import glob
import os
from sklearn.model_selection import train_test_split
import numpy as np

base_dir = sys.argv[1]

all_country_path_list = ["Japan/train/annotations/xmls",
                  "India/train/annotations/xmls",
                  "Czech/train/annotations/xmls",
                  "Norway/train/annotations/xmls",
                  "United_States/train/annotations/xmls",
]

all_file_list = []

for country_path in all_country_path_list:
    file_list = sorted(glob.glob(os.path.join(base_dir, country_path) + "/*.xml"))
    all_file_list += file_list



X = np.arange(len(all_file_list))
train_ids, test_ids, train_file_list, test_file_list = train_test_split(X, all_file_list, test_size=0.2, random_state=42)
print("Number of all files: ", len(all_file_list))
print("Number of train files: ", len(train_file_list))
print("Number of test files: ", len(test_file_list))

with open(os.path.join(base_dir, 'all_annotation_path.txt'),'w') as output_file: 
    for file_path in all_file_list:
        output_file.write(file_path + "\n")
with open(os.path.join(base_dir, 'train_annotation_path.txt'),'w') as output_file: 
    for file_path in train_file_list:
        output_file.write(file_path + "\n")
with open(os.path.join(base_dir, 'val_annotation_path.txt'),'w') as output_file: 
    for file_path in test_file_list:
        output_file.write(file_path + "\n")
np.save(os.path.join(base_dir, 'train_id.npy'), train_ids)
np.save(os.path.join(base_dir, 'val_id.npy'), test_ids)