#!/bin/bash

# ./check_local_outputs.sh :: post pre-processing checkup to test if the preprocessing results exist
# Run your script in your local laptop only

# Define the base directory containing all subject directories
BASE_DIR="$1"

# Array of required file paths for each subject
REQUIRED_FILES=(
    "CVR_MAPS/Thresholded/{SUBJECT_ID}_BOLD_CVR_resized_thresholded.nii.gz"
    "rsBOLD_ET/{SUBJECT_ID}_rsBOLD_ET.nii.gz"
    "rsBOLD_ET_Flip/{SUBJECT_ID}_rsBOLD_ET_Flip.nii.gz"
    "T1/{SUBJECT_ID}_T1.nii.gz"
)

# Log files for missing and complete subjects
MISSING_LOG="missing_files.log"
COMPLETE_SUBJECTS_LOG="complete_subjects.log"
> "$MISSING_LOG"  # Clear the missing files log
> "$COMPLETE_SUBJECTS_LOG"  # Clear the complete subjects log

# Check each subject's files
for SUBJECT_DIR in "$BASE_DIR"/*; do
    if [ -d "$SUBJECT_DIR" ]; then
        SUBJECT_ID=$(basename "$SUBJECT_DIR")
        echo "Checking files for subject: $SUBJECT_ID"
        all_files_present=true

        # Check each required file for the subject
        for REL_PATH in "${REQUIRED_FILES[@]}"; do
            FILE_PATH="${SUBJECT_DIR}/${REL_PATH//\{SUBJECT_ID\}/$SUBJECT_ID}"
            
            if [ ! -f "$FILE_PATH" ]; then
                echo "Missing file for subject $SUBJECT_ID: $FILE_PATH" | tee -a "$MISSING_LOG"
                all_files_present=false
            fi
        done

        # Log subject if all files are present
        if [ "$all_files_present" = true ]; then
            echo "$SUBJECT_ID" | tee -a "$COMPLETE_SUBJECTS_LOG"
        fi
    fi
done

# Summary message
if [ -s "$MISSING_LOG" ]; then
    echo "Some files are missing. See $MISSING_LOG for details."
else
    echo "All required files are present for each subject."
fi

echo "Subjects with all files are listed in $COMPLETE_SUBJECTS_LOG."

# Define the full list of possible subjects by listing the files and extracting subject IDs
RAW_DATA_BASE_DIR="../../../../../network_drive"    # raw data folder
all_pos_subs=$(ls ${RAW_DATA_BASE_DIR}/CVR_MAPS | sed -E 's/(SF_[0-9]+)_BOLD_CVR\.nii\.gz/\1/')

# Read COMPLETE_SUBJECTS_LOG into an array using a while loop
completed_subjects=()
while IFS= read -r line; do
    completed_subjects+=("$line")
done < "$COMPLETE_SUBJECTS_LOG"

# Convert `all_pos_subs` string into an array
all_subjects=($all_pos_subs)

# Initialize an empty array for missing subjects
missing_subjects=()

# Check each subject in `all_subjects` to see if it's in `completed_subjects`
for subject in "${all_subjects[@]}"; do
    if [[ ! " ${completed_subjects[@]} " =~ " ${subject} " ]]; then
        missing_subjects+=("$subject")
    fi
done

# Output missing subjects
if [ ${#missing_subjects[@]} -eq 0 ]; then
    echo "All subjects are accounted for in COMPLETE_SUBJECTS_LOG."
else
    echo "The following subjects are missing from COMPLETE_SUBJECTS_LOG:"
    printf '%s\n' "${missing_subjects[@]}"
fi
