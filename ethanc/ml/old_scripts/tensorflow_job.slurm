#! /bin/bash
# ====================================
#SBATCH --job-name=tf-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100
# ====================================

source ~/software/init-conda
conda activate gpu_tflow

python tensorflow_test.py
nvidia-smi
