#!/bin/zsh
#SBATCH --job-name=CVR_ARRAY
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=2GB
##SBATCH --mail-type=END
##SBATCH --mail-user=muhammad.mahajna@ucalgary.ca
#SBATCH --output=slurm_logs/output/fmri_pre/OutputFromCVRJob_%A_%a.out
#SBATCH --error=slurm_logs/error/fmri_pre/ErrorFromCVRJob_%A_%a.err

echo "Hi, let's start working..."

# Resolve base directories to absolute paths
SCRIPT_DIR=$(realpath "${SLURM_SUBMIT_DIR}/../preprocessing")
IN_BASE_DIR=$(realpath "../../../../data/cvr")
OUTPUT_BASE_DIR=$IN_BASE_DIR

echo "Script directory resolved to: $SCRIPT_DIR"
echo "Input base directory resolved to: $IN_BASE_DIR"

# Load the Conda environment
echo "Loading Conda environment and preparing..."
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate cvr_env

# Get the list of subjects
SUBJECT_LIST=($(ls -d $IN_BASE_DIR/*/ | xargs -n 1 basename))

# Check that the SUBJECT_LIST is not empty
if [ ${#SUBJECT_LIST[@]} -eq 0 ]; then
    echo "Error: No subjects found in $IN_BASE_DIR"
    exit 1
fi

echo "Going to process the following subjects: ${SUBJECT_LIST[@]}"

# Allow for testing without SLURM_ARRAY_TASK_ID
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Select the subject based on the job array index
SUBJECT_ID=${SUBJECT_LIST[$SLURM_ARRAY_TASK_ID+1]}

# Check if SUBJECT_ID is set (in case SLURM_ARRAY_TASK_ID exceeds SUBJECT_LIST size)
if [ -z "$SUBJECT_ID" ]; then
    echo "No subject found for SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID. Exiting."
    exit 1
fi

# Run the processing script for the selected subject
echo "Processing subject: $SUBJECT_ID"
PREPROCESS_SCRIPT="${SCRIPT_DIR}/preprocess_raw_data.sh"

if [ ! -f "$PREPROCESS_SCRIPT" ]; then
    echo "Error: Preprocessing script not found at $PREPROCESS_SCRIPT"
    exit 1
fi

bash "$PREPROCESS_SCRIPT" "$IN_BASE_DIR" "$OUTPUT_BASE_DIR" "$SUBJECT_ID" "$SCRIPT_DIR"

echo "Finished processing subject: $SUBJECT_ID"
