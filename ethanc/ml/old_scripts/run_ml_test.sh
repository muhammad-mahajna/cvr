#!/bin/bash
#SBATCH --job-name=Thursday_morning_test_no_image
#SBATCH --output=Small_subset_test_%j.txt
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB

source ~/software/init-conda
conda activate gpu_tflow

python testing_keras_model.py
