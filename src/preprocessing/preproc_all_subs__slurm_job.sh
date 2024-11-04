#!/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=CVR_Array
#SBATCH --array=0-9 ## test this on 10 subject so I don't get the server bombarded
##SBATCH --mail-type=END
##SBATCH --mail-user=muhammad.mahajna@ucalgary.ca
#SBATCH --output=OutputFromCVRJob_%A_%a.out    # %A is the job array ID, %a is the job array index
#SBATCH --error=ErrorFromCVRJob_%A_%a.err      # Standard error

sleep 1s
echo "Running job array task ID: $SLURM_ARRAY_TASK_ID"

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate cvr_env

# Set the base directories
IN_BASE_DIR="../../../../data/cvr"
OUTPUT_BASE_DIR=$IN_BASE_DIR

# Get the list of subjects
SUBJECTS=($(ls -d $IN_BASE_DIR/*/))  # Adjust this line if needed to match the directory structure

# Select the current subject based on the array index
CURRENT_SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

echo "Processing subject: $CURRENT_SUBJECT"

# Run the preprocessing script for the current subject
./preprocess_raw_data.sh "$CURRENT_SUBJECT"

echo "Finished processing subject: $CURRENT_SUBJECT"
