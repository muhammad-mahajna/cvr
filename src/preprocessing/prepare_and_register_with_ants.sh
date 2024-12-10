#!/bin/bash

# prepare_and_register_with_ants.sh :: prepare and register fMRI data to T1 anatomical images

# Example usage:
# ./prepare_and_register_with_ants.sh /path/to/base_dir SUBJECT001
# ./prepare_and_register_with_ants.sh /path/to/base_dir all

echo "-----------------------------------------"
echo "Running prepare_and_register_with_ants script"

# Check for correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <base_directory> <subject_id> or 'all' to process all subjects"
    exit 1
fi

# Resolve the base directory to an absolute path
BASE_DIR=$(realpath "$1")
SUBJECT_ID="$2"

# Function to prepare and register data for a single subject
prepare_and_register_subject() {
    SUBJECT_ID="$1"
    INPUT_DIR="$BASE_DIR/$SUBJECT_ID/preprocessing_results"
    OUTPUT_DIR="$BASE_DIR/$SUBJECT_ID/ants_results"
    LOG_FILE="$OUTPUT_DIR/${SUBJECT_ID}_ants_log.txt"

    mkdir -p "$OUTPUT_DIR"

    echo "Starting ANTs preparation and registration for subject $SUBJECT_ID at $(date)" 

    # Define paths to input files
    T1_IMAGE="$INPUT_DIR/${SUBJECT_ID}_strip.nii.gz"
    FMRI_IMAGE="$INPUT_DIR/rsBOLD_field_corrected.nii.gz"
    FIRST_FRAME_IMAGE="$OUTPUT_DIR/${SUBJECT_ID}_first_frame.nii.gz"

    # Check for required input files
    if [ ! -f "$T1_IMAGE" ]; then
        echo "Error: T1-weighted skull-stripped image not found for subject $SUBJECT_ID: $T1_IMAGE" 
        return 1
    fi
    if [ ! -f "$FMRI_IMAGE" ]; then
        echo "Error: Field-corrected fMRI image not found for subject $SUBJECT_ID: $FMRI_IMAGE" 
        return 1
    fi

    # Extract the first frame from the fMRI data
    if [ ! -f "$FIRST_FRAME_IMAGE" ]; then
        echo "Extracting first frame from field-corrected fMRI data for subject $SUBJECT_ID..." 
        fslroi "$FMRI_IMAGE" "$FIRST_FRAME_IMAGE" 0 1 || {
            echo "Error: Failed to extract first frame for subject $SUBJECT_ID" 
            return 1
        }
    else
        echo "First frame already exists. Skipping extraction." 
    fi

    # Perform ANTs registration
    echo "Running ANTs registration for subject $SUBJECT_ID..." 
    antsRegistrationSyNQuick.sh -d 3 \
        -f "$T1_IMAGE" \
        -m "$FIRST_FRAME_IMAGE" \
        -o "$OUTPUT_DIR/${SUBJECT_ID}_ants_" \
        -t r || {
            echo "Error: ANTs registration failed for subject $SUBJECT_ID" 
            return 1
        }

    # Apply transformation to normalize the fMRI data
    echo "Applying transformation to normalize fMRI data for subject $SUBJECT_ID..." 
    antsApplyTransforms -d 3 -e 3 \
        -i "$FMRI_IMAGE" \
        -r "$FIRST_FRAME_IMAGE" \
        -o "$OUTPUT_DIR/${SUBJECT_ID}_fMRI_normalized.nii.gz" \
        -t "$OUTPUT_DIR/${SUBJECT_ID}_ants_0GenericAffine.mat" \
        -v 2>> "$LOG_FILE" || {
            echo "Error: Transformation failed for subject $SUBJECT_ID" 
            return 1
        }

    echo "ANTs normalization complete for subject $SUBJECT_ID at $(date)" 
}

# Process all subjects or a single subject
if [ "$SUBJECT_ID" == "all" ]; then
    for SUBJECT_DIR in "$BASE_DIR"/*; do
        if [ -d "$SUBJECT_DIR/preprocessing_results" ]; then
            CURRENT_SUBJECT_ID=$(basename "$SUBJECT_DIR")
            prepare_and_register_subject "$CURRENT_SUBJECT_ID" || {
                echo "Error: Processing failed for subject $CURRENT_SUBJECT_ID" 
                exit 1
            }
        fi
    done
else
    # Process a single subject
    prepare_and_register_subject "$SUBJECT_ID" || {
        echo "Error: Processing failed for subject $SUBJECT_ID" 
        exit 1
    }
fi
