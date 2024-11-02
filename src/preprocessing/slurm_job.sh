#!/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=CVR
##SBATCH --partition=cpu2023
#SBATCH --mail-type=END
#SBATCH --mail-user=muhammad.mahajna@ucalgary.ca
#SBATCH --output=OutputFromMyCVRJob_%j.out
#SBATCH --error=ErrorFromMyCVRJob_%j.err    # Standard error

sleep 1s
echo Hello World

eval "$(~/miniconda3/bin/conda shell.bash hook)"

conda activate cvr_env

IN_BASE_DIR="../../../../data/cvr"
OUTPUT_BASE_DIR=$IN_BASE_DIR

./preprocess_raw_data.sh 

echo Finished model training
