#!/bin/bash
# mkdir -p bdd100k

ZIP_DIR="/home/phd_li/dataset/r2s100k"


for zipfile in "$ZIP_DIR"/*.zip; do
    echo "Processing: $zipfile"
    unzip $zipfile -d /home/phd_li/dataset/r2s100k/
done