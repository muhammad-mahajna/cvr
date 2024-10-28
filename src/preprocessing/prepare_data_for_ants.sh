#!/bin/bash

# Example usage:
# ./prepare_data_for_ants.sh /data/project1 SUBJECT001

# Check for correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <base_directory> <subject_id>"
    exit 1
fi

# Configurable base directories
BASE_DIR="$1"
SUBJECT_ID="$2"
INPUT_DIR="$BASE_DIR/RAW_DATA/completed/$SUBJECT_ID"
OUTPUT_DIR="$BASE_DIR/ANTS_RESULTS/$SUBJECT_ID"
LOG_FILE="$OUTPUT_DIR/${SUBJECT_ID}_ants_log.txt"

# Check if the subject directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Directory not found: $INPUT_DIR"
    exit 1
fi

# Create output directory for ANTs results
mkdir -p "$OUTPUT_DIR"

# Initialize log file
echo "Starting ANTs preparation for subject $SUBJECT_ID at $(date)" > "$LOG_FILE"

# Use the skull-stripped T1-weighted image from previous steps
if [ ! -f "$OUTPUT_DIR/${SUBJECT_ID}_brain.nii.gz" ]; then
    echo "Using skull-stripped T1 image from previous preprocessing for subject $SUBJECT_ID..." | tee -a "$LOG_FILE"
    cp "$INPUT_DIR/${SUBJECT_ID}_brain.nii.gz" "$OUTPUT_DIR/"
else
    echo "T1 image already present in output directory. Skipping copy." | tee -a "$LOG_FILE"
fi

# Use the motion-corrected fMRI data from previous steps
if [ ! -f "$OUTPUT_DIR/${SUBJECT_ID}_fMRI_mc.nii.gz" ]; then
    echo "Using motion-corrected fMRI data from previous preprocessing for subject $SUBJECT_ID..." | tee -a "$LOG_FILE"
    cp "$INPUT_DIR/rsBOLD_fc.nii.gz" "$OUTPUT_DIR/${SUBJECT_ID}_fMRI_mc.nii.gz"
else
    echo "Motion-corrected fMRI data already present in output directory. Skipping copy." | tee -a "$LOG_FILE"
fi

# Extract the first frame from the motion-corrected fMRI data
if [ ! -f "$OUTPUT_DIR/${SUBJECT_ID}_first_frame.nii.gz" ]; then
    echo "Extracting first frame from motion-corrected fMRI data for subject $SUBJECT_ID..." | tee -a "$LOG_FILE"
    fslroi "$OUTPUT_DIR/${SUBJECT_ID}_fMRI_mc.nii.gz" "$OUTPUT_DIR/${SUBJECT_ID}_first_frame" 0 1 || { echo "Error extracting first frame for $SUBJECT_ID" | tee -a "$LOG_FILE"; exit 1; }
else
    echo "First frame already extracted. Skipping step." | tee -a "$LOG_FILE"
fi

# Perform ANTs normalization
echo "Running ANTs normalization for subject $SUBJECT_ID..." | tee -a "$LOG_FILE"
antsRegistrationSyNQuick.sh -d 3 \
    -f "$BASE_DIR/templates/MNI152_T1_1mm.nii.gz" \
    -m "$OUTPUT_DIR/${SUBJECT_ID}_brain.nii.gz" \
    -o "$OUTPUT_DIR/${SUBJECT_ID}_ants_" || { echo "Error during ANTs registration for $SUBJECT_ID" | tee -a "$LOG_FILE"; exit 1; }

echo "ANTs normalization complete for subject $SUBJECT_ID at $(date)" | tee -a "$LOG_FILE"
