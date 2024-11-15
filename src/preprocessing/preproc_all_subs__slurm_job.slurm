#!/bin/zsh
#SBATCH --job-name=CVR_ARRAY
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=2GB
##SBATCH --mail-type=END
##SBATCH --mail-user=muhammad.mahajna@ucalgary.ca
#SBATCH --output=OutputFromCVRJob_%A_%a.out
#SBATCH --error=ErrorFromCVRJob_%A_%a.err

echo "Hi, let's start working..."

# Define base directories
IN_BASE_DIR="../../../../data/cvr"
OUTPUT_BASE_DIR=$IN_BASE_DIR

echo "Loading Conda environment and preparing..."

# Load the conda environment
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate cvr_env

# Get the list of subjects by listing directories in IN_BASE_DIR
SUBJECT_LIST=($(ls -d $IN_BASE_DIR/*/ | xargs -n 1 basename))

# Check that the SUBJECT_LIST is not empty
if [ ${#SUBJECT_LIST[@]} -eq 0 ]; then
    echo "Error: No subjects found in $IN_BASE_DIR"
    exit 1
fi

echo "Going to process the following subjects: ${SUBJECT_LIST[@]}"

# Select the subject based on the job array index
SUBJECT_ID=${SUBJECT_LIST[$SLURM_ARRAY_TASK_ID+1]}

# Check if SUBJECT_ID is set (in case SLURM_ARRAY_TASK_ID exceeds SUBJECT_LIST size)
if [ -z "$SUBJECT_ID" ]; then
    echo "No subject found for SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID. Exiting."
    exit 1
fi

# Run the processing script for the selected subject
echo "Processing subject: $SUBJECT_ID"
./preprocess_raw_data.sh "$IN_BASE_DIR" "$OUTPUT_BASE_DIR" "$SUBJECT_ID"

echo "Finished processing subject: $SUBJECT_ID"
