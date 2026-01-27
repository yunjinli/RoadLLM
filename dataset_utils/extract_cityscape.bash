#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <ZIP_DIR> <OUTPUT_DIR>"
    exit 1
fi

ZIP_DIR="$1"
ZIP_DIR="${1%/}"
OUTPUT_DIR="$2"
OUTPUT_DIR="${2%/}"

echo "Extracting Cityscape dataset from $ZIP_DIR to $OUTPUT_DIR"

zipfiles=("leftImg8bit_trainextra.zip" "leftImg8bit_trainvaltest.zip" "gtFine_trainvaltest.zip")

mkdir -p $OUTPUT_DIR

for str in ${zipfiles[@]}; do
    echo "Processing: $ZIP_DIR/$str"
    unzip $ZIP_DIR/$str -d $OUTPUT_DIR
done