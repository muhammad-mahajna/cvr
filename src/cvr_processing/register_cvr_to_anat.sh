#!/bin/bash

# Directory containing the thresholded CVR maps
INPUT_DIR="$HOME/CVR_MAPS/processed_cvr_maps"
# Directory to store registered CVR maps
OUTPUT_DIR="$INPUT_DIR/registered"

# Load necessary FSL and ANTs modules
echo "Loading FSL and ANTs modules..."
module load fsl
module load ants

# Create output directory if it doesn't exist
echo "Setting up output directory at: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Process each subject's thresholded CVR map
echo "Starting registration of CVR maps to anatomical images..."
for cvr_map in "$INPUT_DIR"/*_thresholded.nii.gz; do
    # Extract subject ID from the filename
    subject_id=$(basename "$cvr_map" | cut -d '_' -f1)
    echo "-----------------------------------------"
    echo "Registering CVR map for subject: $subject_id"

    # Define paths for preprocessed T1 and CVR images
    T1_IMAGE="$HOME/anat_processed/${subject_id}_T1_brain.nii.gz"
    OUTPUT_PREFIX="$OUTPUT_DIR/${subject_id}_CVR_registered_"

    # Check if T1-weighted image exists
    if [ ! -f "$T1_IMAGE" ]; then
        echo "Warning: T1-weighted image for subject $subject_id not found at $T1_IMAGE. Skipping."
        continue
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
done

echo "-----------------------------------------"
echo "All CVR maps have been processed and registered to anatomical images."
echo "Registered maps are available in: $OUTPUT_DIR"
