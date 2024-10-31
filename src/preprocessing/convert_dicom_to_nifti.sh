#!/bin/bash
# Script to convert DICOM files to NIfTI format for multiple subjects using dcm2niix.
# This script finds each subject’s specified subfolder and converts its DICOM files to NIfTI.

# Run this command to make the script executable: 
# chmod +x ./convert_dicom_to_nifti.sh

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <base_dir> <nifti_dir> <search_term>"
    exit 1
fi

# Define input parameters
BASE_DIR="$1"      # Base directory where all subject folders are located
NIFTI_DIR="$2"     # Directory where the converted NIfTI files will be saved
SEARCH_TERM="$3"   # Subfolder name to search for within each subject folder

# Log the input parameters for reference
echo "Base directory (DICOM): $BASE_DIR"
echo "NIfTI output directory: $NIFTI_DIR"
echo "Searching for subfolders containing: $SEARCH_TERM"

# Create output directory if it doesn't exist
mkdir -p "$NIFTI_DIR"

# Start timestamp
START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
echo "Conversion started at: $START_TIME"

# Convert DICOM files to NIfTI format for each subject's specified subfolder
for subject_path in "$BASE_DIR"/*; do
    if [ -d "$subject_path" ]; then
        subject_name=$(basename "$subject_path")
        
        # Define the expected NIfTI output file path
        output_nifti_file="$NIFTI_DIR/${subject_name}.nii.gz"

        # Check if the NIfTI file already exists
        if [ -f "$output_nifti_file" ]; then
            echo "NIfTI file for subject $subject_name already exists. Skipping conversion."
            continue
        fi
        
        # Find the specified subfolder within the subject's directory
        target_folder=$(find "$subject_path" -type d -name "*$SEARCH_TERM*" | head -n 1)

        if [ -z "$target_folder" ]; then
            echo "Warning: No '$SEARCH_TERM' folder found for subject $subject_name. Skipping."
            continue
        fi

        echo "Converting DICOM files for subject $subject_name from $target_folder..."

        # Run conversion using dcm2niix and name output file after subject ID
        dcm2niix -z y -f "${subject_name}" -o "$NIFTI_DIR" "$target_folder"
        
        # Check if the conversion was successful
        if [ $? -eq 0 ]; then
            echo "Conversion successful for subject $subject_name."
        else
            echo "Error during conversion for subject $subject_name."
        fi
    fi
done

# End timestamp
END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
echo "Conversion completed at: $END_TIME"
echo "DICOM to NIfTI conversion complete for all subjects."
