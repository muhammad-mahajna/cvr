#!/bin/bash

# Directory containing the original CVR maps
RAW_DATA_BASE_DIR="$1"
INPUT_DIR="$RAW_DATA_BASE_DIR/CVR_MAPS"

# Base directory for output
BASE_DIR="$2"

# Threshold value for processing
THRESHOLD_VALUE=0.7

# Load necessary FSL module
echo "Loading FSL module..."

# Start processing each CVR map
echo "Starting CVR map processing in directory: $INPUT_DIR"
for file in "$INPUT_DIR"/*_BOLD_CVR.nii.gz; do
    # Extract subject ID from the filename
    filename=$(basename "$file")
    SUBJECT_ID=$(echo "$filename" | sed -E 's/(SF_[0-9]+)_BOLD_CVR\.nii\.gz/\1/')
    
    # Define the subject-specific output directory
    SUBJECT_OUTPUT_DIR="$BASE_DIR/$SUBJECT_ID/CVR_MAPS"
    
    # Define subdirectories for BASE, Resized, and Thresholded files within the subject's folder
    BASE_SUBDIR="$SUBJECT_OUTPUT_DIR/BASE"
    RESIZED_DIR="$SUBJECT_OUTPUT_DIR/Resized"
    THRESHOLD_DIR="$SUBJECT_OUTPUT_DIR/Thresholded"
    mkdir -p "$BASE_SUBDIR" "$RESIZED_DIR" "$THRESHOLD_DIR"
    
    echo "-----------------------------------------"
    echo "Processing CVR map for subject: $SUBJECT_ID"
    
    # Copy the original input file to the BASE subdirectory
    cp "$file" "$BASE_SUBDIR/"
    echo "Input file copied to: $BASE_SUBDIR/"

    # Resize the CVR map
    resized_file="$RESIZED_DIR/${SUBJECT_ID}_BOLD_CVR_resized.nii.gz"
    echo "Resizing map for subject $SUBJECT_ID..."
    fslroi "$file" "$resized_file" 0 -1 0 -1 6 26
    echo "Resized map saved to: $resized_file"
    
    # Apply threshold to the resized map
    thresholded_file="$THRESHOLD_DIR/${SUBJECT_ID}_BOLD_CVR_resized_thresholded.nii.gz"
    echo "Applying threshold of $THRESHOLD_VALUE for subject $SUBJECT_ID..."
    fslmaths "$resized_file" -uthr "$THRESHOLD_VALUE" "$thresholded_file"
    echo "Thresholded map saved to: $thresholded_file"
done

echo "-----------------------------------------"
echo "All CVR maps have been processed and saved in their respective subject directories under $BASE_DIR."
