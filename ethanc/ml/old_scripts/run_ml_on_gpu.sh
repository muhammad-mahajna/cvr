#!/bin/bash

#SBATCH --job-name=ML_GPU_test1
#SBATCH --output=ML_GPU_output_%j.txt
#SBATCH --mail-type=END
#SBATCH --mail-user=ethan.church@ucalgary.ca
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100
#SBATCH --mem=150GB


source ~/software/init-conda
conda activate gpu_tflow

python for_arc.py
nvidia-smi
