#!/bin/bash
#SBATCH --job-name=install_conda
#SBATCH --output=install_conda.log
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# Step 1: Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/Miniconda3-latest-Linux-x86_64.sh

# Step 2: Run the installer
bash ~/Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3

# Step 3: Initialize Conda
~/miniconda3/bin/conda init

# Step 4: Reload shell to apply changes
source ~/.bashrc

# Confirm installation
conda --version

echo "Conda installation complete and initialized."
