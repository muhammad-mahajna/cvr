#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <input_base_directory> <output_base_directory> <subject_id> <script_dir>"
    exit 1
fi

# Get the input and output base directories, subject ID, and script directory
IN_BASE_DIR="$1"
OUTPUT_BASE_DIR="$2"
SUBJECT_ID="$3"
SCRIPT_DIR="$4"

# Check if the subject directory exists
if [ ! -d "$IN_BASE_DIR/$SUBJECT_ID" ]; then
    echo "Subject directory not found: $IN_BASE_DIR/$SUBJECT_ID"
    exit 1
fi

echo "Processing subject: $SUBJECT_ID"

# Run each preprocessing step for the specific subject
bash "$SCRIPT_DIR/preprocess_fmri_anatomical.sh" "$IN_BASE_DIR" "$OUTPUT_BASE_DIR" "$SUBJECT_ID" "$SCRIPT_DIR"
bash "$SCRIPT_DIR/prepare_and_register_with_ants.sh" "$IN_BASE_DIR" "$SUBJECT_ID" "$SCRIPT_DIR"

# Uncomment if CVR preprocessing is required
# bash "$SCRIPT_DIR/preprocess_ref_cvr_data.sh" "$IN_BASE_DIR" "$OUTPUT_BASE_DIR" "$SUBJECT_ID" "$SCRIPT_DIR"

echo "Preprocessing complete for subject: $SUBJECT_ID"
