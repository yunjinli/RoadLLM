#!/bin/bash
ZIP_DIR="/home/phd_li/dataset/"
OUTPUT_DIR="/home/phd_li/dataset/cityscapes"
zipfiles=("leftImg8bit_trainextra.zip" "leftImg8bit_trainvaltest.zip" "gtFine_trainvaltest.zip" "gtCoarse.zip")

mkdir -p $OUTPUT_DIR

for str in ${zipfiles[@]}; do
    echo "Processing: $ZIP_DIR$str"
    unzip $ZIP_DIR$str -d $OUTPUT_DIR
done