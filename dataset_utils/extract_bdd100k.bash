#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <ZIP_DIR> <OUTPUT_DIR>"
    exit 1
fi

ZIP_DIR="$1"
ZIP_DIR="${1%/}"
OUTPUT_DIR="$2"
OUTPUT_DIR="${2%/}"
mkdir -p "$OUTPUT_DIR"

for zipfile in "$ZIP_DIR"/*.zip; do
    echo "Processing: $zipfile"
    unzip $zipfile "bdd100k/*" -d "$OUTPUT_DIR"
done