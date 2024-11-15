#!/bin/bash

# Define the base directory containing all subject directories
IN_BASE_DIR="../../../../data/cvr"

# Count the number of subjects by listing the directories
SUBJECT_COUNT=$(ls -d "$IN_BASE_DIR"/*/ | wc -l)

# Ensure there are subjects to process
if [ "$SUBJECT_COUNT" -eq 0 ]; then
    echo "No subjects found in $IN_BASE_DIR."
    exit 1
fi

echo "Submitting array job with $SUBJECT_COUNT tasks (one for each subject)..."

# Submit the array job with dynamic array range
JOB_ID=$(sbatch --array=0-$(($SUBJECT_COUNT - 1)) preproc_all_subs__slurm_job.slurm | awk '{print $4}')

# Submit the post-processing job with a dependency on the completion of the array job
sbatch --dependency=afterok:$JOB_ID post_processing_checkup.sh

echo "Post-processing job submitted with dependency on Job ID: $JOB_ID"
