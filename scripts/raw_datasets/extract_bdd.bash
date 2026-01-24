#!/bin/bash
# mkdir -p bdd100k

ZIP_DIR="/home/phd_li/dataset"


for zipfile in "$ZIP_DIR"/*.zip; do
    echo "Processing: $zipfile"
    unzip $zipfile "bdd100k/*" -d /home/phd_li/dataset/
done