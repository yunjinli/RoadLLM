#!/bin/bash
ZIP_DIR="nuimages-v1.0-all-samples.tgz"
OUTPUT_DIR="/home/phd_li/dataset/nuimages"

mkdir -p $OUTPUT_DIR

tar -xzvf "$ZIP_DIR" -C "$OUTPUT_DIR" --wildcards '*/CAM_FRONT/*'
# META_DIR="nuimages-v1.0-all-metadata.tgz"
# tar -zxvf "$META_DIR" -C "$OUTPUT_DIR"