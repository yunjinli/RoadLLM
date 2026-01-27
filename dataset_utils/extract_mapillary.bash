#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <ZIP_DIR> <OUTPUT_DIR>"
    exit 1
fi

ZIP_DIR="$1"
OUTPUT_DIR="$2"

mkdir -p $OUTPUT_DIR

unzip $ZIP_DIR "*/v2.0/*" "*/images/*" -d $OUTPUT_DIR