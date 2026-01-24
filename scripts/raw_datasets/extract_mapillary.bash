#!/bin/bash
ZIP_DIR="/home/phd_li/dataset/An-ZjB1Zm61yAZG0ozTymz8I8NqI4x0MrYrh26dq7kPgfu8vf9ImrdaOAVOFYbJ2pNAgUnVGBmbue9lTgdBOb5BbKXIpFs0fpYWqACbrQDChAA2fdX0zS9PcHu7fY8c-FOvyBVxPNYNFQuM.zip"
OUTPUT_DIR="/home/phd_li/dataset/mapillary"

mkdir -p $OUTPUT_DIR

unzip $ZIP_DIR "*/v2.0/*" "*/images/*" -d $OUTPUT_DIR