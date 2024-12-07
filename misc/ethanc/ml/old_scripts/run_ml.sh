#!/bin/bash 
#SBATCH --job-name=ML_test2
#SBATCH --output=ML_output_%j.txt
#SBATCH --time=72:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB

source ~/software/init-conda
conda activate gpu_tflow 

python for_arc_cpu.py



