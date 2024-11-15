#!/bin/bash
#SBATCH --job-name=CVR_PostProcessCheck
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --mem=1GB
#SBATCH --output=PostProcessCheck_%j.out
#SBATCH --error=PostProcessCheck_%j.err

# Define the base directory containing all subject directories
BASE_DIR="../../../../data/cvr"

# Array of required file paths for each subject
REQUIRED_FILES=(
    "ants_results/{SUBJECT_ID}_fMRI_normalized.nii.gz"
    "CVR_MAPS/Thresholded/{SUBJECT_ID}_BOLD_CVR_resized_thresholded.nii.gz"
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
