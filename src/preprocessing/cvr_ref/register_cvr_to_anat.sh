#!/bin/bash

# register_cvr_to_anat.sh :: Register reference CVR maps to the anatomical image (T1)

# Check if correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <BASE_DIR> <SUBJECT_ID> or 'all' to process all subjects"
    exit 1
fi

# Set BASE_DIR and SUBJECT_ID from input arguments
BASE_DIR="$1"
SUBJECT_ID="$2"

# Directory suffix to store registered CVR maps for each subject
OUTPUT_DIR_SUFFIX="registered"

# Function to process a single subject
process_subject() {
    local SUBJECT_ID="$1"
    local INPUT_DIR="$BASE_DIR/$SUBJECT_ID/CVR_MAPS/Thresholded"
    local OUTPUT_DIR="$BASE_DIR/$SUBJECT_ID/CVR_MAPS/$OUTPUT_DIR_SUFFIX"
    
    # Find the thresholded CVR map for the subject
    CVR_MAP=$(find "$INPUT_DIR" -name "${SUBJECT_ID}_BOLD_CVR_resized_thresholded.nii.gz" 2>/dev/null)
    if [ -z "$CVR_MAP" ]; then
        echo "Warning: Thresholded CVR map not found for subject $SUBJECT_ID in $INPUT_DIR. Skipping."
        echo $CVR_MAP
        return 1
    fi

    # Create output directory if it doesn't exist
    echo "Setting up output directory for subject $SUBJECT_ID at: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"

    echo "-----------------------------------------"
    echo "Registering CVR map for subject: $SUBJECT_ID"

    # Define paths to T1 image and output prefix
    T1_IMAGE="$BASE_DIR/$SUBJECT_ID/preprocessing_results/${SUBJECT_ID}_strip.nii.gz"
    OUTPUT_PREFIX="$OUTPUT_DIR/${SUBJECT_ID}_CVR_registered_"

    # Check if T1-weighted image exists
    if [ ! -f "$T1_IMAGE" ]; then
        echo "Warning: T1-weighted image for subject $SUBJECT_ID not found at $T1_IMAGE. Skipping."
        return 1
    fi

    echo "Using T1-weighted image: $T1_IMAGE"
    echo "Using CVR map: $CVR_MAP"

    # Run ANTs registration
    echo "Running ANTs registration for subject $SUBJECT_ID..."
    antsRegistrationSyNQuick.sh -d 3 -f "$T1_IMAGE" -m "$CVR_MAP" -o "$OUTPUT_PREFIX" -t r

    FIRST_FRAME_IMAGE="$BASE_DIR/$SUBJECT_ID/ants_results/${SUBJECT_ID}_first_frame.nii.gz"
    AFFINE="$BASE_DIR/$SUBJECT_ID/ants_results/${SUBJECT_ID}_ants_0GenericAffine.mat"

    echo "Running ANTs transform for subject $SUBJECT_ID..."

    antsApplyTransforms -d 3 \
        -i $CVR_MAP \
        -r $FIRST_FRAME_IMAGE \
        -o "$OUTPUT_DIR/${SUBJECT_ID}_CVR_normalized.nii.gz" \
        -t $AFFINE
    

    # Verify if registration output was created
    if [ -f "${OUTPUT_PREFIX}Warped.nii.gz" ]; then
        echo "Registration successful for subject: $SUBJECT_ID"
        echo "Output saved at: ${OUTPUT_PREFIX}Warped.nii.gz"
    else
        echo "Error: Registration failed for subject $SUBJECT_ID. Output file not found."
    fi
}

# Check if processing all subjects or a single subject
if [ "$SUBJECT_ID" == "all" ]; then
    echo "Processing all subjects in $BASE_DIR"
    for subject_dir in "$BASE_DIR"/*; do
        if [ -d "$subject_dir" ]; then
            current_SUBJECT_ID=$(basename "$subject_dir")
            process_subject "$current_SUBJECT_ID"
        fi
    done
else
    echo "Processing single subject: $SUBJECT_ID"
    process_subject "$SUBJECT_ID"
fi

echo "-----------------------------------------"
echo "All specified CVR maps have been processed and registered to anatomical images."
echo "Registered maps are available in each subject's respective 'registered' folder."
