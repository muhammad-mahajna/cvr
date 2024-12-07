#!/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=23:00:00
#SBATCH --mem=36GB
#SBATCH --job-name=CVR_MODEL
#SBATCH --mail-type=END
#SBATCH --mail-user=muhammad.mahajna@ucalgary.ca
#SBATCH --output=OutputFromMyFirstJob_%j.out
#SBATCH --error=ErrorFromMyFirstJob_%j.err    # Standard error

sleep 1s
echo Hi!
echo "Let's get going."
echo Load Conda Env and get things ready

# Load the conda environment
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate cvr_env

echo Starting model training

time python -u train_model.py

echo Finished model training
