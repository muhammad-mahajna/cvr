#!/bin/bash

# Example usage:
# ./prepare_and_register_with_ants.sh /data/project1 SUBJECT001
# ./prepare_and_register_with_ants.sh /data/project1 all

# Check for correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <base_directory> <subject_id> or 'all' to process all subjects"
    exit 1
fi

# Configurable base directories
BASE_DIR="$1"
SUBJECT_ID="$2"

# Detect system and set MNI template path based on environment
if [[ "$HOSTNAME" == "arc" ]]; then
    FSL_DIR="/home/muhammad.mahajna/workspace/software/fsl"
else
    FSL_DIR="/Users/muhammadmahajna/workspace/software/fsl"
fi
MNI_TEMPLATE="${FSL_DIR}/data/standard/MNI152_T1_1mm.nii.gz"


# Function to prepare and register data for a single subject
prepare_and_register_subject() {
    SUBJECT_ID="$1"
    INPUT_DIR="$BASE_DIR/$SUBJECT_ID/preprocessing_results"
    OUTPUT_DIR="$BASE_DIR/$SUBJECT_ID/ants_results"
    LOG_FILE="$OUTPUT_DIR/${SUBJECT_ID}_ants_log.txt"

    # Check if the subject directory exists
    if [ ! -d "$INPUT_DIR" ]; then
        echo "Directory not found: $INPUT_DIR"
        return 1
    fi

    # Create output directory for ANTs results if it doesn't exist
    mkdir -p "$OUTPUT_DIR"

    # Initialize log file
    echo "Starting ANTs preparation and registration for subject $SUBJECT_ID at $(date)" > "$LOG_FILE"

    # Define paths to input files
    T1_IMAGE="$INPUT_DIR/${SUBJECT_ID}_strip.nii.gz"
    FMRI_IMAGE="$INPUT_DIR/rsBOLD_field_corrected.nii.gz"
    FIRST_FRAME_IMAGE="$OUTPUT_DIR/${SUBJECT_ID}_first_frame.nii.gz"

    # Check if required files are available
    if [ ! -f "$T1_IMAGE" ]; then
        echo "T1-weighted skull-stripped image not found for subject $SUBJECT_ID: $T1_IMAGE" | tee -a "$LOG_FILE"
        return 1
    fi
    if [ ! -f "$FMRI_IMAGE" ]; then
        echo "Field-corrected fMRI image not found for subject $SUBJECT_ID: $FMRI_IMAGE" | tee -a "$LOG_FILE"
        return 1
    fi

    # Extract the first frame from the field-corrected fMRI data if not already done
    if [ ! -f "$FIRST_FRAME_IMAGE" ]; then
        echo "Extracting first frame from field-corrected fMRI data for subject $SUBJECT_ID..." | tee -a "$LOG_FILE"
        fslroi "$FMRI_IMAGE" "$FIRST_FRAME_IMAGE" 0 1 || { echo "Error extracting first frame for $SUBJECT_ID" | tee -a "$LOG_FILE"; return 1; }
    else
        echo "First frame already extracted. Skipping step." | tee -a "$LOG_FILE"
    fi

    # Perform ANTs normalization directly on the skull-stripped T1 image
    echo "Running ANTs normalization for subject $SUBJECT_ID..." | tee -a "$LOG_FILE"
    antsRegistrationSyNQuick.sh -d 3 \
        -f "$MNI_TEMPLATE" \
        -m "$T1_IMAGE" \
        -o "$OUTPUT_DIR/${SUBJECT_ID}_ants_" || { echo "Error during ANTs registration for $SUBJECT_ID" | tee -a "$LOG_FILE"; return 1; }

    echo "ANTs normalization complete for subject $SUBJECT_ID at $(date)" | tee -a "$LOG_FILE"
}

# If the subject ID is 'all', process all subjects in the directory
if [ "$SUBJECT_ID" == "all" ]; then
    for SUBJECT_DIR in "$BASE_DIR"/preprocessing_results/*; do
        if [ -d "$SUBJECT_DIR" ]; then
            CURRENT_SUBJECT_ID=$(basename "$SUBJECT_DIR")
            prepare_and_register_subject "$CURRENT_SUBJECT_ID" || {
                echo "Error processing subject $CURRENT_SUBJECT_ID"
                exit 1
            }
        fi
    done
else
    # Process a single subject
    prepare_and_register_subject "$SUBJECT_ID"
fi
