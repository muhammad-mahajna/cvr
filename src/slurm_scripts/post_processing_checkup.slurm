#!/bin/bash
#SBATCH --job-name=CVR_PostProcessCheck
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --mem=1GB
#SBATCH --output=slurm_logs/output/post_proc/PostProcessCheck_%j.out
#SBATCH --error=slurm_logs/error/post_proc/PostProcessCheck_%j.err

# Ensure SLURM log directories exist
mkdir -p slurm_logs/output/post_proc slurm_logs/error/post_proc

echo "Starting post-processing checkup..."

# Resolve base directories to absolute paths
SCRIPT_DIR=$(realpath $(dirname "$0"))  # Path to the script directory
BASE_DIR=$(realpath "${SCRIPT_DIR}/../../../../data/cvr")
FMRI_INPUT_DIR="${BASE_DIR}/ml/fmri_input"
CVR_REF_DIR="${BASE_DIR}/ml/cvr_ref"

# Log files for missing and complete subjects
MISSING_LOG="${SCRIPT_DIR}/output/missing_files.log"
COMPLETE_SUBJECTS_LOG="${SCRIPT_DIR}/output/complete_subjects.log"
> "$MISSING_LOG"  # Clear the missing files log
> "$COMPLETE_SUBJECTS_LOG"  # Clear the complete subjects log

# Array of required file paths for each subject
REQUIRED_FILES=(
    "ants_results/{SUBJECT_ID}_fMRI_normalized.nii.gz"
    "CVR_MAPS/Thresholded/{SUBJECT_ID}_BOLD_CVR_resized_thresholded.nii.gz"
)

# Ensure output directories exist
mkdir -p "$FMRI_INPUT_DIR" "$CVR_REF_DIR"

# Check each subject's files
echo "Checking files in base directory: $BASE_DIR"
for SUBJECT_DIR in "$BASE_DIR"/*; do
    if [ -d "$SUBJECT_DIR" ]; then
        SUBJECT_ID=$(basename "$SUBJECT_DIR")
        echo "Checking files for subject: $SUBJECT_ID"
        all_files_present=true

        # Define file paths
        fMRI_FILE="$SUBJECT_DIR/ants_results/${SUBJECT_ID}_fMRI_normalized.nii.gz"
        CVR_FILE="$SUBJECT_DIR/CVR_MAPS/Thresholded/${SUBJECT_ID}_BOLD_CVR_resized_thresholded.nii.gz"

        # Check each required file for the subject
        if [ ! -f "$fMRI_FILE" ]; then
            echo "Missing file for subject $SUBJECT_ID: $fMRI_FILE" | tee -a "$MISSING_LOG"
            all_files_present=false
        fi
        if [ ! -f "$CVR_FILE" ]; then
            echo "Missing file for subject $SUBJECT_ID: $CVR_FILE" | tee -a "$MISSING_LOG"
            all_files_present=false
        fi

        # Log subject if all files are present and copy files
        if [ "$all_files_present" = true ]; then
            echo "$SUBJECT_ID" | tee -a "$COMPLETE_SUBJECTS_LOG"

            # Copy the fMRI normalized file to fmri_input directory
            cp "$fMRI_FILE" "$FMRI_INPUT_DIR/${SUBJECT_ID}_fMRI_normalized.nii.gz"
            echo "Copied $fMRI_FILE to $FMRI_INPUT_DIR"

            # Copy the CVR thresholded file to cvr_ref directory
            cp "$CVR_FILE" "$CVR_REF_DIR/${SUBJECT_ID}_BOLD_CVR_resized_thresholded.nii.gz"
            echo "Copied $CVR_FILE to $CVR_REF_DIR"
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
