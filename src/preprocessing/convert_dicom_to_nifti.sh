#!/bin/bash
# Script to convert DICOM files to NIfTI format for multiple subjects or a single subject using dcm2niix.

# Check if the correct number of arguments is provided
if [ "$#" -lt 5 ] || [ "$#" -gt 6 ]; then
    echo "Usage: $0 <dicom_dir> <base_out_dir> <NIFTI_BASE_DIR> <search_term> <file_suffix> [<subject_id>]"
    exit 1
fi

# Define input parameters
BASE_DICOM_DIR="$1"
NIFTI_BASE_DIR="$2"
OUT_SUB_DIR="$3"
SEARCH_TERM="$4"
FILE_SUFFIX="$5"
SUBJECT_ID="$6"  # Optional subject ID to process a single subject

# Log the input parameters for reference
echo "Base directory (DICOM): $BASE_DICOM_DIR"
echo "NIfTI output directory: $NIFTI_BASE_DIR"
echo "Searching for subfolders containing: $SEARCH_TERM"
echo "Using $FILE_SUFFIX as a file name suffix"
[ -n "$SUBJECT_ID" ] && echo "Processing only for subject: $SUBJECT_ID" || echo "Processing all subjects"

# Create output directory if it doesn't exist
mkdir -p "$NIFTI_BASE_DIR"

# Start timestamp
START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
echo "Conversion started at: $START_TIME"

# Function to process a single subject
convert_subject() {
    local subject_path="$1"
    local subject_name="$2"

    # Define the expected NIfTI output file path
    local output_nifti_file="$NIFTI_BASE_DIR/$subject_name/$OUT_SUB_DIR/${subject_name}_${FILE_SUFFIX}.nii.gz"

    # Check if the NIfTI file already exists
    if [ -f "$output_nifti_file" ]; then
        echo "NIfTI file for subject $subject_name already exists. Skipping conversion."
        return
    fi

    mkdir -p "$NIFTI_BASE_DIR/$subject_name/$OUT_SUB_DIR/"

    # Find the specified subfolder within the subject's directory
    local target_folder
    target_folder=$(find "$subject_path" -type d -name "*-$SEARCH_TERM" | head -n 1)

    if [ -z "$target_folder" ]; then
        target_folder=$(find "$subject_path" -type d -name "*-$SEARCH_TERM*" | head -n 1)
        if [ -z "$target_folder" ]; then
            echo "Warning: No '$SEARCH_TERM' folder found for subject $subject_name. Skipping."
            return
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
}

# Convert DICOM files to NIfTI format for each subject
if [ -n "$SUBJECT_ID" ]; then
    # Process single specified subject
    subject_path="$BASE_DICOM_DIR/$SUBJECT_ID"
    if [ -d "$subject_path" ]; then
        subject_name=$(basename "$subject_path" | sed -E 's/.*-(SF|sf)_?([0-9]+)/SF_\2/' | tr '[:lower:]' '[:upper:]')
        convert_subject "$subject_path" "$subject_name"
    else
        echo "Error: Directory for subject $SUBJECT_ID not found in $BASE_DICOM_DIR."
        exit 1
    fi
else
    # Process all subjects in the directory
    for subject_path in "$BASE_DICOM_DIR"/*; do
        if [ -d "$subject_path" ]; then
            subject_name=$(basename "$subject_path" | sed -E 's/.*-(SF|sf)_?([0-9]+)/SF_\2/' | tr '[:lower:]' '[:upper:]')
            convert_subject "$subject_path" "$subject_name"
        fi
    done
fi

# End timestamp
END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
echo "Conversion completed at: $END_TIME"
echo "DICOM to NIfTI conversion complete."
