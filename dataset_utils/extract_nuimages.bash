#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <ZIP_SAMPLES> <ZIP_METADATA> <OUTPUT_DIR>"
    exit 1
fi

ZIP_SAMPLES="$1"
ZIP_METADATA="$2"
OUTPUT_DIR="$3"
OUTPUT_DIR="${3%/}"

mkdir -p $OUTPUT_DIR

tar -xzvf "$ZIP_SAMPLES" -C "$OUTPUT_DIR" --wildcards '*/CAM_FRONT/*'
tar -zxvf "$ZIP_METADATA" -C "$OUTPUT_DIR"