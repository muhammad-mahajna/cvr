# Data PreProcessing Pipeline for fMRI and Anatomical Data

## Summary of Steps with Purpose

1. **Convert raw DICOM data** using `convert_dicom_to_nifti.sh`.
   - **Purpose**: Standardize raw data into the NIfTI format for easier handling.

2. **Preprocess fMRI and anatomical data** using `batch_preprocess_fmri_anat.sh`.
   - **Purpose**: Perform motion correction, skull stripping, and topup distortion correction to clean and prepare the data. The processing is automated for multiple subjects to ensure consistency and reduce manual intervention.

3. **Optionally, prepare data for ANTs** using `prepare_data_for_ants.sh`.
   - **Purpose**: Get data ready for normalization using ANTs by performing motion correction and skull stripping.

4. **Normalize and extract ROI time series** using `normalize_and_extract_roi.sh`.
   - **Purpose**: Align functional and anatomical images to a template and extract time series data from regions of interest.

5. **Preprocess CVR data** using `preprocess_cvr_data.sh`.
   - **Purpose**: Handle preprocessing specific to CVR studies, including motion correction for specialized datasets.

6. **Run functional analysis** using `run_functional_analysis.sh`.
   - **Purpose**: Perform statistical analysis on functional MRI data using the FSL `feat` tool, setting up design files and automating the analysis.


## Step-by-Step Process

### 1. Convert DICOM to NIfTI
- **Script**: `convert_dicom_to_nifti.sh`
- **Purpose**: Convert raw DICOM files from the scanner to the more accessible NIfTI format. This is essential for standardizing the data format for subsequent preprocessing and analysis.
```bash
./convert_dicom_to_nifti.sh <dicom_dir> <nifti_dir> <number_of_subjects>"
```

### 2. Preprocess fMRI and Anatomical Data
- **Script**: `batch_preprocess_fmri_anat.sh`
- **Purpose**: Perform preprocessing on both functional (fMRI) and anatomical images. This step includes skull stripping the T1-weighted anatomical images, applying motion correction to fMRI data, and using topup to correct for susceptibility-induced distortions. It ensures the data is clean and ready for further analysis.
Automate the preprocessing of multiple subjects by applying the preprocess_fmri_anat.sh script across all subjects in a directory. This reduces manual workload and ensures consistency across subject data.

```bash
./batch_preprocess_fmri_anat.sh <directory_name> <subject_id>
```

### 3. Prepare Data for ANTs (Optional)
- **Script**: `prepare_data_for_ants.sh`
- **Purpose**: Prepare data for further normalization using ANTs (Advanced Normalization Tools). This step involves converting DICOM files, skull stripping, and motion correction to set up images for more advanced image registration processes using ANTs.

```bash
./prepare_data_for_ants.sh <directory_name> <subject_id>
```

### 5. Normalize and Extract ROI Time Series
- **Script**: `normalize_and_extract_roi.sh`
- **Purpose**: Normalize both functional and anatomical data to a standard template (using ANTs) and extract mean time series data from specific regions of interest (ROIs). This helps to align the subject's brain images to a common space and then analyze specific regions for further study.

```bash
./normalize_and_extract_roi.sh
```

### 6. Preprocess CVR Data (Specialized)
- **Script**: `preprocess_cvr_data.sh`
- **Purpose**: Handle the preprocessing of Cerebrovascular Reactivity (CVR) data, including motion correction and other adjustments specific to CVR studies. This step is specialized for datasets where CO2/O2 challenges are used, ensuring the fMRI data is prepped for CVR analysis.

```bash
./preprocess_cvr_data.sh
```

### 7. Run Functional Analysis (FSL FEAT)
- **Script**: `run_functional_analysis.sh`
- **Purpose**: Run functional MRI analysis using FSL’s feat tool, which performs various statistical analyses on the preprocessed functional data. The script automates this process by setting up design files and running the analysis for each subject.

```bash
./run_functional_analysis.sh
```


# Preprocessing pipeline:
This pipeline will prepare all data (fMRI+Anatomical+CVR)
1. Make sure that data exists in local folders 
2. run `./prepare_raw_data.sh` locally to create working data and upload this data to the cluster.
3. run `./submit_all_preprocessing_jobs.sh.slurm` in the cluster to perform all preprocessing steps on all available subjects (inlcuding rsBOLD and reference CVR data analysis)