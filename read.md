# Estimating CVR using ML   

## Overview

This project aims to develop a machine-learning model for estimating CVR directly from rsBOLD fMRI data. By integrating neural network architectures (e.g. CNN, RNN), the model will capture temporal patterns associated with CVR in the rsBOLD signals. The ultimate objective is to create a robust tool that produces accurate CVR maps while eliminating the challenges associated with traditional CVR measurement methods.

The project uses `PyTorch` as the deep learning framework,  and is designed to run efficiently on a GPU cluster using Slurm.

## Project Structure
Data is expected be stored in the following directory or under `/home/muhammad.mahajna/workspace/research/data/cvr_est_project`

```
├── env_utils/                                  # Folder containing environment setup scripts. 
├── misc/                                       # Folder containing temporary code that was tried. There is no guarantee that this code will work
├── src/                                        # Source code folder
|   ├── model                                   # ML model folder
|   |   ├── data                                # Folder containing temporary data
|   |   ├── misc                                # Folder containing temporary code that was tried. There is no guarantee that this code will work
|   |   ├── CVRRegressionModel.py               # Python file containg the main CVR ML model (pytorch)
|   |   ├── model_design.png/svg                # A visual representation of the CVR ML model
|   |   ├── model_training_notebook.ipynb       # Jupetyr notebook containing the model training code.
|   |   ├── model_testing_notebook.ipynb        # Jupetyr notebook containing the model testing code.
|   |   ├── train_model.py                      # Python file for training the ML model. This file is used in cluster training.
|   |   └── train_model_job.sh                  # Script for submitting the model training script. TODO: move to the slrum folder
|   ├── preprocessing                           # Folder containing all preprocessing scripts
|   |   ├── cvr_ref                             # Folder CVR preprocessing steps
|   |   |   |── process_cvr_maps.sh             # Scripts for running CVR preprocesisng steps.
|   |   |   └── register_cvr_to_anat.sh         # Scripts for running CVR registration tasks
|   |   ├── check_local_outputs.sh              # Scripts that checks if all outputs exists in the local folder
|   |   ├── convert_dicom_to_nifti.sh           # Scripts for converting DICOM files to NIFTII files
|   |   ├── prepare_and_register_with_ants.sh   # Scripts for preparing fMRI file and running the registration
|   |   ├── prepare_raw_data.sh                 # Scripts for preparing the raw data. It runrs the DICOM conversion script for all  data types, checks the outputs and uploads them to ARC
|   |   ├── preprocess_fmri_anatomical.sh       # Scripts for running the fMRI preprocessing tasks. Don't run this script directly.
|   |   ├── preprocess_raw_data.sh              # Scripts for running all fMRI preprocessing tasks
|   |   └── preprocess_ref_cvr_data.sh          # Scripts for running all reference CVR preprocessing tasks.
│   └── slurm_scripts                           # Folder containing SLURM scripts used in the project. 
|       ├── submit_all_preprocessing_jobs.sh    # submits all preprocessing steps using sbatch. This is the only job that you need to submit as it takes care of all preprocessing steps automatically.
|       ├── preproc_all_subs.slurm              # Script for running the preprocessing steps for all subjects
|       ├── register_cvr_all_subs.slurm         # Script for running the cvr registration steps for all subjects
|       └── post_processing_checkup.slurm       # Script for running post processing checkup steps
├── setup_conda_environment.sh                  # Script to set up the conda environment
├── model_notebook.py                           # Jupyter notebook for training and evaluating the model
├── train_model.py                              # Python script for training the model (same code as the notebook)
├── slurm_job_gpu.sh                            # Slurm submission script (using GPU)
├── slurm_job_cpu.sh                            # Slurm submission script (using CPU)
└── misc/                                       # Folder containing temporary code that was tried. There is no guarantee that this code will work but it gives an overview of the methdds that was tried. 

```

## Setup Instructions

### Prerequisites

- [Anaconda/Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed.
- Access to a GPU-enabled server or cluster (recommended)

### Step 1: Set up Conda Environment

1. **Create and activate the conda environment** using the provided script:

   ```bash
   ./conda_env_setup.sh
   ./prep_cluster_env.sh
   ```

   These scripts create an conda environment named `cvr_env`,  and installs all necessary packages.
   Alternatively, if you have all the required libraries already installed, you can skip this step. 

2. **Verify the installation**:

   ```bash
   conda activate mm_enel645_assg2
   conda list
   ```

### Step 2: Train the Model

1. **Submit the training job to the cluster** using the provided Slurm script:

   ```bash
   sbatch slurm_job_gpu.sh
   ```

   This Slurm script (`slurm_job_gpu.sh`) allocates the necessary resources (nodes, CPUs, GPUs, memory, etc.) and runs the `train_model.py` script to start training the model.

2. **Monitor the job**:

   You can monitor the Slurm job with:

   ```bash
   squeue -u your_username
   ```

   You can also check the output and error logs using the `.out` and `.err` files generated by Slurm.

### Step 3: Model Evaluation

After training completes, the model will be evaluated on the test set. The results are displayed in a confusion matrix, along with other performance metrics like accuracy, precision, and recall.

## Files Description

- **`setup_conda_environment.sh`**: Bash script to set up the conda environment and install all dependencies.
- **`train_model.py`**: Python script containing all necessary code to train the multimodal model.
- **`slurm_job_gpu/cpu.sh`**: Slurm batch script for submitting the training job to a cluster.
- **`train_image_descriptions.csv`**: CSV file that contains descriptions/captions for the training images.

## Requirements

- `torch`: For deep learning model creation.
- `torchvision`: For image data processing.
- `transformers`: For text data processing with DistilBERT.
- `scikit-learn`: For evaluation metrics.
- `matplotlib` and `seaborn`: For plotting confusion matrix and other visualizations.
- `pillow` and `numpy`: For image and numerical data handling.

## Example Usage

1. **Run the training script locally**:

   ```bash
   python train_model.py
   ```

2. **Submit the job to a cluster**:

   ```bash
   sbatch slurm_job_gpu.sh
   ```

3. **Check the output** in the Slurm `.out` file to monitor the progress and performance metrics.

## Data and Methods

This project uses a combination of image data and text descriptions to classify images into four categories: Black, Blue, Green, and TTR. The model architecture consists of a ResNet50 model for image feature extraction and a DistilBERT model for processing text descriptions.
A fully connected neural network combines the features and classifies input data into one of the said output classes. 

### Dataset

- **Image Data**: RGB images of varying resolutions.
- **Text Data**: Captions associated with each image.

### Methodology

1. **Data Preprocessing**: Resizing images, tokenizing text, and applying necessary augmentations.
2. **Model**: A multimodal neural network with ResNet50 and DistilBERT components.
3. **Training**: Uses a weighted cross-entropy loss function to account for class imbalance.
4. **Evaluation**: Performance metrics like accuracy, precision, recall, and confusion matrix are computed.


Now the fun part, enjoy!