#!/bin/bash

# Load specific FSL and ANTs modules for testing
echo "Loading FSL and ANTs modules..."
module load fsl/6.0.5
module load ants

# Define paths for the test subject’s T1 and rsBOLD images
TEST_SUBJECT_ID="SF_01035"
T1_IMAGE="$HOME/anat_processed/${TEST_SUBJECT_ID}_T1_brain.nii.gz"
FUNC_IMAGE="$HOME/func/unprocessed/${TEST_SUBJECT_ID}/${TEST_SUBJECT_ID}_rsBOLD.nii.gz"
OUTPUT_DIR="$HOME/test_registration_output"

# Create output directory if it doesn't exist
echo "Setting up output directory at: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Check if required images exist
if [ ! -f "$T1_IMAGE" ]; then
    echo "Error: T1-weighted image for test subject $TEST_SUBJECT_ID not found at $T1_IMAGE. Exiting."
    exit 1
fi
if [ ! -f "$FUNC_IMAGE" ]; then
    echo "Error: rsBOLD image for test subject $TEST_SUBJECT_ID not found at $FUNC_IMAGE. Exiting."
    exit 1
fi

echo "Using T1-weighted image: $T1_IMAGE"
echo "Using rsBOLD image: $FUNC_IMAGE"

# Extract mean volume from rsBOLD scan for testing
echo "Extracting mean volume from rsBOLD scan for subject $TEST_SUBJECT_ID..."
fslmaths "$FUNC_IMAGE" -Tmean "$OUTPUT_DIR/${TEST_SUBJECT_ID}_rsBOLD_mean.nii.gz"

# Run ANTs registration for the test subject
echo "Running ANTs registration on mean rsBOLD image for subject $TEST_SUBJECT_ID..."
antsRegistrationSyNQuick.sh -d 3 -f "$T1_IMAGE" -m "$OUTPUT_DIR/${TEST_SUBJECT_ID}_rsBOLD_mean.nii.gz" -o "$OUTPUT_DIR/${TEST_SUBJECT_ID}_test_registration_"

# Apply the computed transformation to the full rsBOLD image
echo "Applying transformations to full rsBOLD scan for subject $TEST_SUBJECT_ID..."
antsApplyTransforms -d 3 -e 3 -i "$FUNC_IMAGE" -r "$T1_IMAGE" \
    -t "$OUTPUT_DIR/${TEST_SUBJECT_ID}_test_registration_1Warp.nii.gz" \
    -t "$OUTPUT_DIR/${TEST_SUBJECT_ID}_test_registration_0GenericAffine.mat" \
    -o "$OUTPUT_DIR/${TEST_SUBJECT_ID}_rsBOLD_test_registered.nii.gz"

# Verify if the final output was created
if [ -f "$OUTPUT_DIR/${TEST_SUBJECT_ID}_rsBOLD_test_registered.nii.gz" ]; then
    echo "Test registration completed successfully for subject: $TEST_SUBJECT_ID"
    echo "Registered rsBOLD image saved at: $OUTPUT_DIR/${TEST_SUBJECT_ID}_rsBOLD_test_registered.nii.gz"
else
    echo "Error: Test registration failed for subject $TEST_SUBJECT_ID. Output file not found."
fi

echo "-----------------------------------------"
echo "Test registration process completed."
echo "All outputs are available in: $OUTPUT_DIR"
