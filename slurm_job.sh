#!/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=1GB
#SBATCH --job-name=MyFirstJobOnARC
#SBATCH --partition=cpu2023
#SBATCH --mail-type=END
#SBATCH --mail-user=muhammad.mahajna@ucalgary.ca
#SBATCH --output=OutputFromMyFirstJob_%j.out
#SBATCH --error=ErrorFromMyFirstJob_%j.err    # Standard error

sleep 1s
echo Hello World


IN_BASE_DIR="../../../../data/cvr"
OUTPUT_BASE_DIR=$IN_BASE_DIR

./preprocess_fmri_anatomical.sh $IN_BASE_DIR $OUTPUT_BASE_DIR 19797-20220304-SF_01033


echo Finished model training
