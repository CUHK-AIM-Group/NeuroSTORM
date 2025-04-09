#!/bin/bash

INPUT_DIR=$1
OUTPUT_DIR=$2

mkdir -p "$OUTPUT_DIR"


for file in "$INPUT_DIR"/*.nii.gz; do
  filename=$(basename "$file" .nii.gz)
  trimmed_filename=${filename:0:${#filename}-12}
  output_file="${OUTPUT_DIR}/${trimmed_filename}brainmask.nii.gz"
  
  echo "Processing $file -> $output_file"
  bet2 "$file" "$output_file" -f 0.5 -g 0
done
