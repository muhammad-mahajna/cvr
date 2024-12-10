#!/bin/bash

# process_cvr_maps.sh :: preprocess reference CVR maps

# Check if the correct number of arguments is provided
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <raw_data_base_dir> <base_output_dir> [<subject_id>]"
    exit 1
fi

# Directory containing the original CVR maps
RAW_DATA_BASE_DIR="$1"
INPUT_DIR="$RAW_DATA_BASE_DIR/CVR_MAPS"

# Base directory for output
BASE_DIR="$2"

# Optional subject ID for processing a single subject
SUBJECT_ID="$3"

# Threshold value for processing
THRESHOLD_VALUE=0.7

# Function to process a single subject's CVR map
process_cvr_map() {
    local file="$1"
    local subject_id="$2"
    
    # Define the subject-specific output directory
    SUBJECT_OUTPUT_DIR="$BASE_DIR/$subject_id/CVR_MAPS"
    
    # Define subdirectories for BASE, Resized, and Thresholded files within the subject's folder
    BASE_SUBDIR="$SUBJECT_OUTPUT_DIR/BASE"
    RESIZED_DIR="$SUBJECT_OUTPUT_DIR/Resized"
    THRESHOLD_DIR="$SUBJECT_OUTPUT_DIR/Thresholded"
    mkdir -p "$BASE_SUBDIR" "$RESIZED_DIR" "$THRESHOLD_DIR"
    
    echo "-----------------------------------------"
    echo "Processing CVR map for subject: $subject_id"
    
    # Copy the original input file to the BASE subdirectory
    cp "$file" "$BASE_SUBDIR/"
    echo "Input file copied to: $BASE_SUBDIR/"

    # Resize the CVR map
    resized_file="$RESIZED_DIR/${subject_id}_BOLD_CVR_resized.nii.gz"
    echo "Resizing map for subject $subject_id..."
    fslroi "$file" "$resized_file" 0 -1 0 -1 6 26
    echo "Resized map saved to: $resized_file"
    
    # Apply threshold to the resized map
    thresholded_file="$THRESHOLD_DIR/${subject_id}_BOLD_CVR_resized_thresholded.nii.gz"
    echo "Applying threshold of $THRESHOLD_VALUE for subject $subject_id..."
    fslmaths "$resized_file" -uthr "$THRESHOLD_VALUE" "$thresholded_file"
    echo "Thresholded map saved to: $thresholded_file"
}

# Start processing
echo "Starting CVR map processing in directory: $INPUT_DIR"

if [ -n "$SUBJECT_ID" ]; then
    # Process a single specified subject
    file="$INPUT_DIR/${SUBJECT_ID}_BOLD_CVR.nii.gz"
    if [ -f "$file" ]; then
        process_cvr_map "$file" "$SUBJECT_ID"
    else
        echo "Error: No file found for subject $SUBJECT_ID in $INPUT_DIR."
        exit 1
    fi
else
    # Process all subjects in the provided directory
    for file in "$INPUT_DIR"/*_BOLD_CVR.nii.gz; do
        # Extract subject ID from the filename
        filename=$(basename "$file")
        subject_id=$(echo "$filename" | sed -E 's/(SF_[0-9]+)_BOLD_CVR\.nii\.gz/\1/')
        
        process_cvr_map "$file" "$subject_id"
    done
fi

echo "-----------------------------------------"
echo "All CVR maps have been processed and saved in their respective subject directories under $BASE_DIR."
