#!/bin/zsh
#SBATCH --job-name=CVR_ARRAY
#SBATCH --array=0-9 ## Test this with 10 subjects
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=80GB
##SBATCH --mail-type=END
##SBATCH --mail-user=muhammad.mahajna@ucalgary.ca
#SBATCH --output=OutputFromCVRJob_%A_%a.out
#SBATCH --error=ErrorFromCVRJob_%A_%a.err


# This script will run preprocessing scrip for all subjects by crating an array of jobs; a job for each subject. 
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

echo "Going to process the following subjects: $SUBJECT_LIST"

# Select the subject based on the job array index
SUBJECT_ID=${SUBJECT_LIST[$SLURM_ARRAY_TASK_ID+1]}

# Run the processing script for the selected subject
./preprocess_raw_data.sh $IN_BASE_DIR $OUTPUT_BASE_DIR $SUBJECT_ID

echo "Finished processing subject: $SUBJECT_ID"
