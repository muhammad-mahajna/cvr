#!/bin/bash
#SBATCH --job-name=Wednesday_Afternoon_test_image_prediction
#SBATCH --output=model_load_run_2_%j.txt
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15GB

source ~/software/init-conda
conda activate gpu_tflow

python Predict_image_model.py