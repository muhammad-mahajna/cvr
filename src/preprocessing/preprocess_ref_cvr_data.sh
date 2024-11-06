#!/bin/bash

# Example Usage:
# ./preprocess_ref_cvr_data.sh /path/to/input /path/to/output SUBJECT001 SUBJECT002
# ./preprocess_ref_cvr_data.sh /path/to/input /path/to/output all

# Check if the correct number of arguments is provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <input_base_dir> <output_base_dir> <subject_id(s)> or 'all' to process all subjects"
    exit 1
fi

# Configurable base directories from script arguments
IN_BASE_DIR="$1"
OUTPUT_BASE_DIR="$2"

# Shift arguments so that the third argument becomes the first subject ID
shift 2

# Function to preprocess a single subject
preprocess_subject() {
    SUBJECT_ID="$1"
    INPUT_ECHO1="$IN_BASE_DIR/$SUBJECT_ID/MBME_CO2_O2/${SUBJECT_ID}_MBME_CO2_O2_e1.nii.gz"
    INPUT_ECHO2="$IN_BASE_DIR/$SUBJECT_ID/MBME_CO2_O2/${SUBJECT_ID}_MBME_CO2_O2_e2.nii.gz"
    OUTPUT_DIR="$OUTPUT_BASE_DIR/$SUBJECT_ID/cvr_preprocessing_results"

    # Check if required NIfTI files exist
    if [ ! -f "$INPUT_ECHO1" ]; then
        echo "Echo1 file not found: $INPUT_ECHO1"
        return 1
    fi
    if [ ! -f "$INPUT_ECHO2" ]; then
        echo "Echo2 file not found: $INPUT_ECHO2"
        return 1
    fi

    echo "Processing subject: $SUBJECT_ID"

    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"

    # Perform motion correction on Echo1 and Echo2
    echo "Performing motion correction on fMRI images for subject $SUBJECT_ID..."
    mcflirt -in "$INPUT_ECHO1" -out "$OUTPUT_DIR/${SUBJECT_ID}_CO2_O2_Echo1_MotionCorrected" || { echo "Error during motion correction for Echo1 of $SUBJECT_ID"; return 1; }
    mcflirt -in "$INPUT_ECHO2" -out "$OUTPUT_DIR/${SUBJECT_ID}_CO2_O2_Echo2_MotionCorrected" || { echo "Error during motion correction for Echo2 of $SUBJECT_ID"; return 1; }

    echo "Preprocessing complete for $SUBJECT_ID."
}

# If the argument is 'all', process all subjects in the input directory
if [ "$1" == "all" ]; then
    for SUBJECT_ID in $(ls "$IN_BASE_DIR"); do
        preprocess_subject "$SUBJECT_ID"
    done
else
    # Process each subject passed as an argument
    for SUBJECT_ID in "$@"; do
        preprocess_subject "$SUBJECT_ID"
    done
fi
