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

SUBJECT_COUNT = 5
echo "Submitting preprocessing array job with $SUBJECT_COUNT tasks..."

# Step 1: Submit the preprocessing array job and capture its Job ID
PREPROCESS_JOB_ID=$(sbatch --array=0-$(($SUBJECT_COUNT - 1)) preproc_all_subs__slurm_job.slurm | awk '{print $4}')
echo "Preprocessing array job submitted with Job ID: $PREPROCESS_JOB_ID"

echo "Submitting registration array job with dependencies on preprocessing jobs..."

# Step 2: Submit each job in the registration array with dependency on the corresponding preprocessing job
for (( i=0; i<$SUBJECT_COUNT; i++ )); do
    sbatch --job-name=CVR_ARRAY_REGISTER \
           --dependency=afterok:${PREPROCESS_JOB_ID}_$i \
           --nodes=1 \
           --ntasks-per-node=1 \
           --cpus-per-task=1 \
           --time=01:00:00 \
           --mem=20GB \
           --output=OutputFromCVRRegisterJob_%A_%a.out \
           --error=ErrorFromCVRRegisterJob_%A_%a.err \
           --export=SLURM_ARRAY_TASK_ID=$i \
           register_cvr_all_subs__slurm_job.slurm
    echo "Submitted registration job $i with dependency on preprocessing job ${PREPROCESS_JOB_ID}_$i"
done

echo "Submitting post-processing checkup job with dependency on all registration jobs..."

# Step 3: Submit the post-processing job, dependent on all registration jobs completing
REGISTER_CVR_JOB_IDS=$(squeue --noheader --format="%i" --name=CVR_ARRAY_REGISTER | paste -sd: -)
sbatch --dependency=afterok:$REGISTER_CVR_JOB_IDS post_processing_checkup.sh
echo "Post-processing checkup job submitted with dependency on registration jobs"
