#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <ZIP_DIR> <OUTPUT_DIR>"
    exit 1
fi
# ZIP_DIR is the directory containing all the zip files
# - test.zip
# - train.zip
# - val.zip
# - test_labels.zip
# - train_labels.zip
# - val_labels.zip

ZIP_DIR="$1"
ZIP_DIR="${1%/}"
OUTPUT_DIR="$2"
OUTPUT_DIR="${2%/}"

for zipfile in "$ZIP_DIR"/*.zip; do
    echo "Processing: $zipfile"
    unzip $zipfile -d $OUTPUT_DIR/
done