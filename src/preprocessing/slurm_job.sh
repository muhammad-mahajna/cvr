#!/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=48GB
#SBATCH --job-name=MyFirstJobOnARC
##SBATCH --partition=cpu2023
#SBATCH --mail-type=END
#SBATCH --mail-user=muhammad.mahajna@ucalgary.ca
#SBATCH --output=OutputFromMyFirstJob_%j.out
#SBATCH --error=ErrorFromMyFirstJob_%j.err    # Standard error

sleep 1s
echo Hello World

conda activate cvr_env

IN_BASE_DIR="../../../../data/cvr"
OUTPUT_BASE_DIR=$IN_BASE_DIR

./preprocess_raw_data.sh 

echo Finished model training
