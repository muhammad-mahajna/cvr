#!/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=23:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=MyFirstJobOnARC
#SBATCH --mail-type=END
#SBATCH --mail-user=muhammad.mahajna@ucalgary.ca
#SBATCH --output=OutputFromMyFirstJob_%j.out
#SBATCH --error=ErrorFromMyFirstJob_%j.err    # Standard error

sleep 1s
echo Hi!
echo Starting model training

time python -u train_model.py

echo Finished model training
