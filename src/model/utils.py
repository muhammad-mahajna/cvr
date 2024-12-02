import nibabel as nib
import numpy as np
import nibabel as nib
import os, re
from scipy.io import savemat
from sklearn.preprocessing import StandardScaler

def load_tf(input):
    img = nib.load(input)
    return img.get_fdata()

def process_file(filepath, is_target=False, remove_time_points=5, data_threshold=0, max_zeros=10, slice_range=None):
    """
    Process an fMRI file for input or target data, with optional slice selection.

    Args:
        filepath (str): Path to the fMRI data file.
        is_target (bool): Whether to process as target data. If True, flattens spatial dimensions.
        remove_time_points (int): Number of initial time points to remove (for inputs only).
        data_threshold (float): Values below this threshold are set to zero (for inputs only).
        max_zeros (int): Maximum number of zero values allowed in a voxel's time series.
        slice_range (list or tuple, optional): Range of slices to select, e.g., [4, 23]. Defaults to None (use all slices).

    Returns:
        numpy.ndarray: Processed data in 2D array format (voxels, time points or 1 for targets).
    """
    data = load_tf(filepath)  # Load the data (4D for inputs, 3D for targets)

    # Select specific slices
    if False and slice_range is not None:
        if is_target:
            data = data[:, :, slice_range[0]:slice_range[1]]  # Target is 3D
        else:
            data = data[:, :, slice_range[0]:slice_range[1], :]  # Input is 4D

    if is_target:
        # Targets: Flatten spatial dimensions to (num_voxels, 1)
        reshaped = data.reshape((-1, 1))

        # Clip the target values to the range [-0.7, 0.7]
        #reshaped = np.clip(reshaped, -0.7, 0.7)
    else:
        # Inputs: Remove initial time points and reshape
        data = data[..., remove_time_points:]  # Slice the time dimension
        reshaped = data.reshape((-1, data.shape[-1]))  # Inputs: (num_voxels, remaining_time_points)
        # Zero values smaller than the threshold
        if data_threshold > 0:
            reshaped[reshaped < data_threshold] = 0

    zero_counts = np.sum(reshaped == 0, axis=1)  # Count zeros along the time dimension
    reshaped[zero_counts > max_zeros, :] = 0

    return reshaped


def normalize_fmri_timeseries(fmri_data):
    """
    Standardizes the fMRI time series for each voxel.

    Args:
        fmri_data (numpy.ndarray): fMRI data of shape (..., T), where T is the time dimension.

    Returns:
        numpy.ndarray: Normalized fMRI data with the same shape as input.
    """
    original_shape = fmri_data.shape
    fmri_data_flat = fmri_data.reshape(-1, original_shape[-1])  # Flatten spatial dimensions
    scaler = StandardScaler()
    fmri_data_normalized = scaler.fit_transform(fmri_data_flat)  # Normalize each time series
    return fmri_data_normalized.reshape(original_shape), scaler


def save_fmri_as_4d_mat(input_path, output_path, remove_time_points=0):
    """
    Saves fMRI data as a 4D .mat file.

    Parameters:
    - input_path (str): Path to the fMRI file (.nii or .nii.gz).
    - output_path (str): Path to save the .mat file.
    - remove_time_points (int): Number of initial time points to remove.

    Returns:
    - None
    """
    # Load fMRI data
    img = nib.load(input_path)
    data = img.get_fdata()  # Shape: (X, Y, Z, T)

    # Remove initial time points if specified
    if remove_time_points > 0:
        data = data[..., remove_time_points:]  # Slice the time dimension

    # Save the data to a .mat file as a 4D array
    mat_dict = {"fmri_4d": data}
    savemat(output_path, mat_dict)

    print(f"4D fMRI data saved to {output_path} with shape {data.shape}")


def save_cvr_as_3d_mat(input_path, output_path):
    
    # Load fMRI data
    img = nib.load(input_path)
    data = img.get_fdata()  # Shape: (X, Y, Z, T)

    
    # Save the data to a .mat file as a 4D array
    mat_dict = {"cvr_3d": data}
    savemat(output_path, mat_dict)

    print(f"3D CVR data saved to {output_path} with shape {data.shape}")


def extract_subject_id(file_path):
    
    # Extract the file name from the path
    file_name = os.path.basename(file_path)  # Extracts 'SF_01135_2_T1.nii.gz'

    # Use a regex pattern to extract the subject ID
    match = re.match(r"(SF_\d+)", file_name)
    if match:
        return match.group(1)
    else:
        raise ValueError("Subject ID not found in the file name.")
    
