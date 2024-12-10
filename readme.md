
# Estimating CVR using ML  

## Overview

This project aims to develop a machine-learning model for estimating CVR directly from rsBOLD fMRI data. By integrating neural network architectures (e.g., CNN, RNN), the model will capture temporal patterns associated with CVR in the rsBOLD signals. The ultimate objective is to create a robust tool that produces accurate CVR maps while eliminating the challenges associated with traditional CVR measurement methods.

The project uses `PyTorch` as the deep learning framework and is designed to run efficiently on a GPU cluster using Slurm.

Data is expected to be stored under `/home/muhammad.mahajna/workspace/research/data/cvr_est_project`

## Project Structure

```
├── env_utils/                                  # Folder containing environment setup scripts. 
├── misc/                                       # Folder containing temporary code that was tried. There is no guarantee that this code will work, but it gives an overview of the methods that were tried. 
├── src/                                        # Source code folder
│   ├── model                                   # ML model folder
│   │   ├── data                                # Folder containing temporary data
│   │   ├── misc                                # Folder containing temporary code that was tried. There is no guarantee that this code will work.
│   │   ├── CVRRegressionModel.py               # Python file containing the main CVR ML model (PyTorch).
│   │   ├── model_design.png/svg                # A visual representation of the CVR ML model.
│   │   ├── model_training_notebook.ipynb       # Jupyter notebook containing the model training code.
│   │   ├── model_testing_notebook.ipynb        # Jupyter notebook containing the model testing code.
│   │   ├── train_model.py                      # Python file for training the ML model. This file is used in cluster training.
│   │   └── train_model_job.sh                  # Script for submitting the model training script. TODO: move to the Slurm folder.
│   ├── preprocessing                           # Folder containing all preprocessing scripts.
│   │   ├── cvr_ref                             # Folder for CVR preprocessing steps.
│   │   │   ├── process_cvr_maps.sh             # Script for running CVR preprocessing steps.
│   │   │   └── register_cvr_to_anat.sh         # Script for running CVR registration tasks.
│   │   ├── check_local_outputs.sh              # Script that checks if all outputs exist in the local folder.
│   │   ├── convert_dicom_to_nifti.sh           # Script for converting DICOM files to NIFTI files.
│   │   ├── prepare_and_register_with_ants.sh   # Script for preparing fMRI files and running the registration.
│   │   ├── prepare_raw_data.sh                 # Script for preparing the raw data. It runs the DICOM conversion script for all data types, checks the outputs, and uploads them to ARC.
│   │   ├── preprocess_fmri_anatomical.sh       # Script for running the fMRI preprocessing tasks. Dont run this script directly.
│   │   ├── preprocess_raw_data.sh              # Script for running all fMRI preprocessing tasks.
│   │   └── preprocess_ref_cvr_data.sh          # Script for running all reference CVR preprocessing tasks.
│   └── slurm_scripts                           # Folder containing SLURM scripts used in the project. 
│       ├── submit_all_preprocessing_jobs.sh    # Submits all preprocessing steps using sbatch. This is the only job that you need to submit as it takes care of all preprocessing steps automatically.
│       ├── preproc_all_subs.slurm              # Script for running the preprocessing steps for all subjects.
│       ├── register_cvr_all_subs.slurm         # Script for running the CVR registration steps for all subjects.
│       └── post_processing_checkup.slurm       # Script for running post-processing checkup steps.
└── readme.md                                   # README file (this file)
```

## Setup Instructions

### Prerequisites

- [Anaconda/Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed.
- Access to a GPU-enabled server or cluster (recommended).

### Step 1: Set Up Conda Environment

1. **Create and activate the Conda environment**:

   ```bash
   ./conda_env_setup.sh
   ./prep_cluster_env.sh
   ```

   These scripts create a Conda environment named `cvr_env` and install all necessary packages.

2. **Verify the installation**:

   ```bash
   conda activate cvr_env
   conda list
   ```

### Step 2: Run Preprocessing tasks
   ```bash
   cd /src/slurm_scripts
   sbatch submit_all_preprocessing_jobs.sh
   ```

### Step 3: Train the Model

1. **Submit the training job**:

   ```bash
   sbatch train_model_job.sh
   ```

2. **Monitor the job**:

   ```bash
   squeue -u your_username
   ```

   Check output and error logs via `.out` and `.err` files.

## Requirements

- `torch`: For deep learning model creation.
- `numpy`: For image and numerical data handling.

## Example Usage

1. **Run the training script locally**:

   ```bash
   python train_model.py
   ```

2. **Submit the job to a cluster**:

   ```bash
   sbatch train_model_job.sh
   ```

3. **Check the output**: under `/slurm_scripts/slurm_logs` locate the `.out` and `.err` files to monitor the progress and performance metrics.

Have fun!