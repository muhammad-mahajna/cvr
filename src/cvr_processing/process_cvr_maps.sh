#!/bin/bash

# Directory containing the original CVR maps
INPUT_DIR="$HOME/CVR_MAPS"
# Directory for storing processed and thresholded CVR maps within INPUT_DIR
OUTPUT_DIR="$INPUT_DIR/processed_cvr_maps"
THRESHOLD_VALUE=0.7

# Load necessary FSL module
echo "Loading FSL module..."
module load fsl

# Create output directory if it doesn't exist
echo "Setting up output directory at: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Start processing each CVR map
echo "Starting CVR map processing in directory: $INPUT_DIR"
for file in "$INPUT_DIR"/*.nii.gz; do
    filename=$(basename "$file")
    echo "-----------------------------------------"
    echo "Processing CVR map: $filename"
    echo "Applying threshold of $THRESHOLD_VALUE..."

    # Threshold and resize CVR map
    output_file="$OUTPUT_DIR/${filename%.*}_thresholded.nii.gz"
    fslmaths "$file" -thr "$THRESHOLD_VALUE" "$output_file"
    
    echo "Thresholded map saved to: $output_file"
done

echo "-----------------------------------------"
echo "All CVR maps have been processed."
echo "Processed files are available in: $OUTPUT_DIR"
