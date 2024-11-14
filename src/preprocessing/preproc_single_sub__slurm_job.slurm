#!/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=80GB
#SBATCH --job-name=CVR
#SBATCH --partition=cpu2023
#SBATCH --mail-type=END
#SBATCH --mail-user=muhammad.mahajna@ucalgary.ca
#SBATCH --output=OutputFromMyCVRJob_%j.out
#SBATCH --error=ErrorFromMyCVRJob_%j.err    # Standard error


# This script will run preprocessing script for a single subject. Select the subject ID below.

echo Hi, lets start working...

# Define base directories
IN_BASE_DIR="../../../../data/cvr"
OUTPUT_BASE_DIR=$IN_BASE_DIR

echo Load Conda Env and get things ready

# Load the conda environment
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate cvr_env

# Get the list of subjects
SUBJECT_LIST=($(ls -d $IN_BASE_DIR/*/ | xargs -n 1 basename))

echo $SUBJECT_LIST

# Select the subject based on the job array index
SUBJECT_ID=${SUBJECT_LIST[7]} # select which subject you want to analyze

echo "Starting analysis for subject: $SUBJECT_ID"

# Run the processing script for the selected subject
./preprocess_raw_data.sh $IN_BASE_DIR $OUTPUT_BASE_DIR $SUBJECT_ID

echo "Finished processing subject: $SUBJECT_ID"