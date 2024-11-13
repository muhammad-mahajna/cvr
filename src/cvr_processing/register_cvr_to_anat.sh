#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <BASE_DIR> <subject_id> or 'all' to process all subjects"
    exit 1
fi

# Set BASE_DIR and SUBJECT_ID from input arguments
BASE_DIR="$1"
SUBJECT_ID="$2"

# Directory suffix to store registered CVR maps for each subject
OUTPUT_DIR_SUFFIX="registered"

# Function to process a single subject
process_subject() {
    local subject_id="$1"
    local input_dir="$BASE_DIR/$subject_id/CVR_MAPS/Thresholded"
    local subject_output_dir="$BASE_DIR/$subject_id/CVR_MAPS/$OUTPUT_DIR_SUFFIX"
    
    # Find the thresholded CVR map for the subject
    cvr_map=$(find "$input_dir" -name "${subject_id}_BOLD_CVR_resized_thresholded.nii.gz" 2>/dev/null)
    if [ -z "$cvr_map" ]; then
        echo "Warning: Thresholded CVR map not found for subject $subject_id in $input_dir. Skipping."
        echo $cvr_map
        return 1
    fi

    # Create output directory if it doesn't exist
    echo "Setting up output directory for subject $subject_id at: $subject_output_dir"
    mkdir -p "$subject_output_dir"

    echo "-----------------------------------------"
    echo "Registering CVR map for subject: $subject_id"

    # Define paths to T1 image and output prefix
    T1_IMAGE="$BASE_DIR/$subject_id/T1/${subject_id}_T1.nii.gz"
    OUTPUT_PREFIX="$subject_output_dir/${subject_id}_CVR_registered_"

    # Check if T1-weighted image exists
    if [ ! -f "$T1_IMAGE" ]; then
        echo "Warning: T1-weighted image for subject $subject_id not found at $T1_IMAGE. Skipping."
        return 1
    fi

    echo "Using T1-weighted image: $T1_IMAGE"
    echo "Using CVR map: $cvr_map"

    # Run ANTs registration
    echo "Running ANTs registration for subject $subject_id..."
    antsRegistrationSyNQuick.sh -d 3 -f "$T1_IMAGE" -m "$cvr_map" -o "$OUTPUT_PREFIX"
    
    # Verify if registration output was created
    if [ -f "${OUTPUT_PREFIX}Warped.nii.gz" ]; then
        echo "Registration successful for subject: $subject_id"
        echo "Output saved at: ${OUTPUT_PREFIX}Warped.nii.gz"
    else
        echo "Error: Registration failed for subject $subject_id. Output file not found."
    fi
}

# Check if processing all subjects or a single subject
if [ "$SUBJECT_ID" == "all" ]; then
    echo "Processing all subjects in $BASE_DIR"
    for subject_dir in "$BASE_DIR"/*; do
        if [ -d "$subject_dir" ]; then
            current_subject_id=$(basename "$subject_dir")
            process_subject "$current_subject_id"
        fi
    done
else
    echo "Processing single subject: $SUBJECT_ID"
    process_subject "$SUBJECT_ID"
fi

echo "-----------------------------------------"
echo "All specified CVR maps have been processed and registered to anatomical images."
echo "Registered maps are available in each subject's respective 'registered' folder."
