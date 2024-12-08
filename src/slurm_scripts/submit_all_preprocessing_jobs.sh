#!/bin/bash

# Get the directory of the current script (to handle relative paths robustly)
SCRIPT_DIR=$(dirname "$0")

# Define the base directory containing all subject directories
IN_BASE_DIR="../../../../data/cvr"

# Count the number of subjects by listing the directories
SUBJECT_COUNT=$(ls -d "$IN_BASE_DIR"/*/ | wc -l)

# Ensure there are subjects to process
if [ "$SUBJECT_COUNT" -eq 0 ]; then
    echo "No subjects found in $IN_BASE_DIR."
    exit 1
fi


echo "Creating SLURM log directories"
mkdir -p slurm_logs/output/fmri_pre/ slurm_logs/error/fmri_pre/
mkdir -p slurm_logs/output/cvr_reg slurm_logs/error/cvr_reg
mkdir -p slurm_logs/output/post_proc slurm_logs/error/post_proc

echo "Submitting preprocessing array job with $SUBJECT_COUNT tasks..."

# Step 1: Submit the preprocessing array job and capture its Job ID
PREPROCESS_JOB_ID=$(sbatch --array=0-$(($SUBJECT_COUNT - 1)) "$SCRIPT_DIR/preproc_all_subs.slurm" | awk '{print $4}')
echo "Preprocessing array job submitted with Job ID: $PREPROCESS_JOB_ID"

echo "Submitting registration array job with dependencies on preprocessing jobs..."

# Step 2: Submit each job in the registration array with dependency on the corresponding preprocessing job
for (( i=0; i<$SUBJECT_COUNT; i++ )); do
    sbatch --job-name=CVR_REG \
           --dependency=afterok:${PREPROCESS_JOB_ID}_$i \
           --export=SLURM_ARRAY_TASK_ID=$i \
           "$SCRIPT_DIR/register_cvr_all_subs.slurm"
    echo "Submitted registration job $i with dependency on preprocessing job ${PREPROCESS_JOB_ID}_$i"
done

echo "Submitting post-processing checkup job with dependency on all registration jobs..."

# Step 3: Submit the post-processing job, dependent on all registration jobs completing
REGISTER_JOB_IDS=$(squeue --noheader --format="%i" --name=CVR_REG | paste -sd: -)
sbatch --dependency=afterok:$REGISTER_JOB_IDS "$SCRIPT_DIR/post_processing_checkup.slurm"
echo "Post-processing checkup job submitted with dependency on registration jobs"
