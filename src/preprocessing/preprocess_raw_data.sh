#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_base_directory> <output_base_directory> <subject_id>"
    exit 1
fi

# Get the input and output base directories and the subject ID from the script arguments
IN_BASE_DIR="$1"
OUTPUT_BASE_DIR="$2"
SUBJECT_ID="$3"

# Check if the subject directory exists
if [ ! -d "$IN_BASE_DIR/$SUBJECT_ID" ]; then
    echo "Subject directory not found: $IN_BASE_DIR/$SUBJECT_ID"
    exit 1
fi

echo "Processing subject: $SUBJECT_ID"

# Run each preprocessing step for the specific subject
./preprocess_fmri_anatomical.sh "$IN_BASE_DIR" "$OUTPUT_BASE_DIR" "$SUBJECT_ID"
./prepare_and_register_with_ants.sh "$IN_BASE_DIR" "$SUBJECT_ID"

# Run CVR preprocessing
#./preprocess_ref_cvr_data.sh "$IN_BASE_DIR" "$OUTPUT_BASE_DIR" "$SUBJECT_ID"

echo "Preprocessing complete for subject: $SUBJECT_ID"
