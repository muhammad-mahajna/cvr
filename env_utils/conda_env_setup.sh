#!/bin/bash
#SBATCH --job-name=setup_cvr_env
#SBATCH --output=setup_cvr_env.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

# Create and activate Conda environment
conda create -n cvr_env python=3.12 -y
source activate cvr_env

# Install packages together to speed up installation
conda install -c conda-forge pytorch torchvision torchaudio cudatoolkit=<cuda_version> fsl ants transformers scikit-learn -y

# Verify installation
python -c "import torch; print(torch.__version__); import ants; print(ants.__version__); import transformers; print(transformers.__version__); import sklearn; print(sklearn.__version__)"
antsRegistration --version
flirt -version  # FSL verification

# Test if the environment is properly set up
echo "Environment setup verification complete."
