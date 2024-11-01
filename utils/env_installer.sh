#!/bin/bash
#SBATCH --job-name=install_libs
#SBATCH --output=install_libs_%j.out
#SBATCH --error=install_libs_%j.err
#SBATCH --time=01:00:00  # Adjust time as necessary
#SBATCH --mem=64G         # Adjust memory as needed
#SBATCH --cpus-per-task=4

# Load necessary modules, if any

# Activate Conda environment
source ~/miniconda3/etc/profile.d/conda.sh  # Adjust to your Conda installation
conda activate FSL

# Install libraries within Conda environment
conda install -y -c conda-forge libgcc-ng=9.3.0
conda install -y -c conda-forge glibc=2.29
conda install -y -c conda-forge ants

echo "Installation of libraries completed."
