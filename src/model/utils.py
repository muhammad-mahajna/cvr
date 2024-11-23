import nibabel as nib
import numpy as np

def load_tf(input):
    img = nib.load(input)
    return img.get_fdata()

def process_file(filepath, is_target=False, remove_time_points=5, data_threshold=0, max_zeros=10):
    # remove_time_points: remove the first remove_time_points because they might be noisy
    data = load_tf(filepath)
    
    if is_target:
        # Targets: Flatten spatial dimensions to (num_voxels, 1)
        reshaped = data.reshape((-1, 1))
    else:
        # Inputs: Remove initial time points and reshape
        data = data[..., remove_time_points:]  # Slice the time dimension
        reshaped = data.reshape((-1, data.shape[-1]))  # Inputs: (num_voxels, remaining_time_points)
        # Zero values smaller than the threshold
        if data_threshold > 0:
            reshaped[reshaped < data_threshold] = 0

    zero_counts = np.sum(reshaped==0, axis=1)  # Count zeros along the time dimension
    reshaped[zero_counts > max_zeros, :] = 0

    return reshaped
