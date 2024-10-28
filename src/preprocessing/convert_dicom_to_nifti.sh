#!/bin/bash
#(Formerly beginning.sh)

# Define base directory (make this configurable)
#BASE_DIR="/home/ethanchurch/TestData"
#NIFTI_DIR="$BASE_DIR/dcm_converted"

#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <base_dir> <nifti_dir> <number_of_subjects>"
    exit 1
fi

# Define input parameters
BASE_DIR="$1"   # Base directory where DICOM files are located
NIFTI_DIR="$2"  # Directory where the converted NIfTI files will be saved
SUBJECTS="$3"   # Number of subjects to process

# Log the input parameters for reference
echo "Base directory (DICOM): $BASE_DIR"
echo "NIfTI output directory: $NIFTI_DIR"
echo "Processing $SUBJECTS subjects."

# Create output directory if it doesn't exist
mkdir -p "$NIFTI_DIR"

# Convert DICOM files to NIfTI format for each subject
for num in $(seq 1 $SUBJECTS); do
    echo "Converting subject $num..."
    # Example of how input DICOM directory is structured:
    # $BASE_DIR/19795-20220304-SF_01030/1-*
    dcm2niix -f '%f_%p' -o "$NIFTI_DIR" "$BASE_DIR/19795-20220304-SF_01030/${num}-*"
    
    # Check if the conversion was successful
    if [ $? -eq 0 ]; then
        echo "Conversion successful for subject $num."
    else
        echo "Error during conversion for subject $num."
    fi
done

echo "DICOM to NIfTI conversion complete for all subjects."
