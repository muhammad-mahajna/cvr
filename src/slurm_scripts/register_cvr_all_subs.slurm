#!/bin/zsh
#SBATCH --job-name=CVR_ARRAY_REGISTER
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=20GB
##SBATCH --mail-type=END
##SBATCH --mail-user=muhammad.mahajna@ucalgary.ca
#SBATCH --output=slurm_logs/output/cvr_reg/OutputFromCVRRegisterJob_%A_%a.out
#SBATCH --error=slurm_logs/error/cvr_reg/ErrorFromCVRRegisterJob_%A_%a.err

echo "Hi, let's start working..."

# Resolve base directories to absolute paths
SCRIPT_DIR=$(realpath $(dirname "$0"))
IN_BASE_DIR=$(realpath "../../../../data/cvr")
OUTPUT_BASE_DIR=$IN_BASE_DIR

echo "Script directory resolved to: $SCRIPT_DIR"
echo "Input base directory resolved to: $IN_BASE_DIR"

# Define the path to the registration script
REGISTER_SCRIPT="${SCRIPT_DIR}/../cvr_processing/register_cvr_to_anat.sh"

# Check if the registration script exists
if [ ! -f "$REGISTER_SCRIPT" ]; then
    echo "Error: Registration script not found at $REGISTER_SCRIPT"
    exit 1
fi

# Load the Conda environment
echo "Loading Conda environment and preparing..."
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate cvr_env

# Get the list of subjects by listing directories in IN_BASE_DIR
SUBJECT_LIST=($(ls -d "$IN_BASE_DIR"/*/ | xargs -n 1 basename))

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

# Run the registration script for the selected subject
echo "Processing subject: $SUBJECT_ID"
bash "$REGISTER_SCRIPT" "$IN_BASE_DIR" "$SUBJECT_ID"

echo "Finished processing subject: $SUBJECT_ID"
