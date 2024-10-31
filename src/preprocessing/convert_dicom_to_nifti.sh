#!/bin/bash
# Script to convert DICOM files to NIfTI format for multiple subjects using dcm2niix.
# This script finds each subject’s specified subfolder and converts its DICOM files to NIfTI.

# Run this command to make the script executable: 
# chmod +x ./convert_dicom_to_nifti.sh

# Check if the correct number of arguments is provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <dicom_dir> <base_out_dir> <NIFTI_BASE_DIR> <search_term> <file_suffix>"
    exit 1
fi

# Define input parameters
BASE_DICOM_DIR="$1"      # Base directory where all subject folders are located
NIFTI_BASE_DIR="$2"     # Directory where the converted NIfTI files will be saved
OUT_SUB_DIR="$3"     # Directory where the converted NIfTI files will be saved
SEARCH_TERM="$4"   # Subfolder name to search for within each subject folder
FILE_SUFFIX="$5"   # Subfolder name to search for within each subject folder

# Log the input parameters for reference
echo "Base directory (DICOM): $BASE_DICOM_DIR"
echo "NIfTI output directory: $NIFTI_BASE_DIR"
echo "Searching for subfolders containing: $SEARCH_TERM"
echo "Using $FILE_SUFFIX as a file name suffix"

# Create output directory if it doesn't exist
mkdir -p "$NIFTI_BASE_DIR"

# Start timestamp
START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
echo "Conversion started at: $START_TIME"

# Convert DICOM files to NIfTI format for each subject's specified subfolder
for subject_path in "$BASE_DICOM_DIR"/*; do
    if [ -d "$subject_path" ]; then
        subject_name=$(basename "$subject_path")
        
        # Define the expected NIfTI output file path
        output_nifti_file="$NIFTI_BASE_DIR/$subject_name/$OUT_SUB_DIR/${subject_name}_${FILE_SUFFIX}.nii.gz"

        # Check if the NIfTI file already exists
        if [ -f "$output_nifti_file" ]; then
            echo "NIfTI file for subject $subject_name already exists. Skipping conversion."
            continue
        fi
        
        mkdir -p "$NIFTI_BASE_DIR/$subject_name/$OUT_SUB_DIR/"

        # Find the specified subfolder within the subject's directory
        target_folder=$(find "$subject_path" -type d -name "*$SEARCH_TERM" | head -n 1)

        if [ -z "$target_folder" ]; then
            target_folder=$(find "$subject_path" -type d -name "*$SEARCH_TERM*" | head -n 1)
            if [ -z "$target_folder" ]; then
                echo "Warning: No '$SEARCH_TERM' folder found for subject $subject_name. Skipping."
                continue
            fi
        fi

        echo "Converting DICOM files for subject $subject_name from $target_folder..."

        # Run conversion using dcm2niix and name output file after subject ID
        dcm2niix -z y -f "${subject_name}_${FILE_SUFFIX}" -o "$NIFTI_BASE_DIR/$subject_name/$OUT_SUB_DIR" "$target_folder"
        
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
