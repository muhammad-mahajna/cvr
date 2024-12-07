#!/bin/bash

# Directory containing unprocessed rsBOLD images
INPUT_DIR="$HOME/func/unprocessed"
# Directory to store completed rsBOLD images
OUTPUT_DIR="$INPUT_DIR/completed"
# Directory containing preprocessed T1 images
T1_DIR="$HOME/anat_processed"
# Directory containing preprocessed motion-corrected rsBOLD files
PREPROCESSED_FUNC_DIR="$HOME/func_preprocessed"

# Load necessary FSL and ANTs modules
echo "Loading FSL and ANTs modules..."
module load fsl
module load ants

# Create output directory if it doesn't exist
echo "Setting up output directory at: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Start processing each subject's rsBOLD image
echo "Starting registration of rsBOLD images to anatomical images..."
for rsbold_dir in "$INPUT_DIR"/*; do
    subject_id=$(basename "$rsbold_dir")
    echo "-----------------------------------------"
    echo "Processing rsBOLD for subject: $subject_id"

    # Define paths for preprocessed T1, rsBOLD, and reference images
    T1_IMAGE="$T1_DIR/${subject_id}_T1_brain.nii.gz"
    FUNC_IMAGE="$PREPROCESSED_FUNC_DIR/${subject_id}_rsBOLD_mc.nii.gz"
    REFERENCE_FRAME="$PREPROCESSED_FUNC_DIR/${subject_id}_rsBOLD_first_frame.nii.gz"
    OUTPUT_PREFIX="$OUTPUT_DIR/${subject_id}_rsBOLD_registered_"

    # Check if the necessary images exist
    if [ ! -f "$T1_IMAGE" ]; then
        echo "Warning: T1-weighted image for subject $subject_id not found at $T1_IMAGE. Skipping."
        continue
    fi
    if [ ! -f "$FUNC_IMAGE" ]; then
        echo "Warning: Motion-corrected rsBOLD image for subject $subject_id not found at $FUNC_IMAGE. Skipping."
        continue
    fi
    if [ ! -f "$REFERENCE_FRAME" ]; then
        echo "Warning: First frame of rsBOLD for subject $subject_id not found at $REFERENCE_FRAME. Skipping."
        continue
    fi

    echo "Using T1-weighted image: $T1_IMAGE"
    echo "Using motion-corrected rsBOLD image: $FUNC_IMAGE"
    echo "Using reference frame: $REFERENCE_FRAME"

    # Run ANTs registration using the reference frame
    echo "Running ANTs registration for subject $subject_id..."
    antsRegistrationSyNQuick.sh -d 3 -f "$T1_IMAGE" -m "$REFERENCE_FRAME" -o "$OUTPUT_PREFIX"

    # Apply transformations to the full rsBOLD scan
    echo "Applying transformations to full rsBOLD scan for subject $subject_id..."
    antsApplyTransforms -d 3 -e 3 -i "$FUNC_IMAGE" -r "$T1_IMAGE" \
        -t "${OUTPUT_PREFIX}1Warp.nii.gz" -t "${OUTPUT_PREFIX}0GenericAffine.mat" \
        -o "$OUTPUT_DIR/${subject_id}_rsBOLD_registered.nii.gz"

    # Verify if registration output was created
    if [ -f "$OUTPUT_DIR/${subject_id}_rsBOLD_registered.nii.gz" ]; then
        echo "Registration successful for rsBOLD image of subject: $subject_id"
        echo "Output saved at: $OUTPUT_DIR/${subject_id}_rsBOLD_registered.nii.gz"
    else
        echo "Error: Registration failed for subject $subject_id. Output file not found."
    fi
done

echo "-----------------------------------------"
echo "All rsBOLD images have been processed and registered to anatomical images."
echo "Registered images are available in: $OUTPUT_DIR"
