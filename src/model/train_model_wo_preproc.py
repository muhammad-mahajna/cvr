# %%
# Import libs
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import nibabel as nib
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from utils import process_file
from utils import normalize_fmri_timeseries 
from CVRRegressionModel import CVRRegressionModel

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')


# %%
# Data loaders

# Preprocessing parameters
REMOVE_TIME_POINT = 0  # Remove the first samples from the fMRI data
SLICE_SELECT = [0, 26] # inclusive of 4, exclusive of 24   
DATA_THRESHOLD = 0
ZERO_COUNT_THRESHOLD = 435 # 10%x430
REMOVE_ALL_ZERO_SAMPS = False
NORMALIZE_DATA = False

# Define dataset for holding the data
class TimeSeriesDataset(Dataset):
    def __init__(self, inputs, targets, normalize=False):
        """
        Dataset class for time-series data.

        Args:
            inputs (numpy.ndarray): Input data of shape (N, T) or higher dimensions (e.g., (X, Y, Z, T)).
            targets (numpy.ndarray): Target data of shape (N, ...) or corresponding spatial dimensions.
            normalize (bool): Whether to normalize the inputs using `normalize_fmri_timeseries`.
        """
        if normalize:
            self.inputs, self.scaler = normalize_fmri_timeseries(inputs)
        else:
            self.inputs = inputs
            self.scaler = None  # No normalization applied

        self.inputs = torch.tensor(self.inputs, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    

# Function for loading the data
def load_dataset_parallel(input_dir, target_dir, num_workers=1, slice_select=SLICE_SELECT):
    """
    Load inputs and targets with optional slice selection using parallel processing.

    Args:
        input_dir (str): Directory containing input files.
        target_dir (str): Directory containing target files.
        num_workers (int): Number of parallel workers.
        slice_select (list or tuple): Range of slices to select (e.g., [4, 24]).

    Returns:
        tuple: (inputs, targets), where both are numpy arrays.
    """

    # List all files in the directories, sorted for consistency
    input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.nii', '.nii.gz'))])
    target_files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(('.nii', '.nii.gz'))])

    # Use parallel processing to load inputs and targets
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        inputs = list(executor.map(
            process_file, 
            input_files, 
            [False] * len(input_files),  # is_target=False for inputs
            [REMOVE_TIME_POINT] * len(input_files),
            [DATA_THRESHOLD] * len(input_files),
            [ZERO_COUNT_THRESHOLD] * len(input_files),
            [slice_select] * len(input_files)  # Pass slice range for inputs
        ))
        targets = list(executor.map(
            process_file, 
            target_files, 
            [True] * len(target_files),  # is_target=True for targets
            [None] * len(target_files),  # remove_time_points not needed for targets
            [0] * len(target_files),  # data_threshold not needed for targets
            [0] * len(target_files),  # max_zeros not needed for targets
            [slice_select] * len(target_files)  # Pass slice range for targets
        ))

    # Stack the loaded data
    inputs = np.vstack([x for x in inputs if x.size > 0]).astype(np.float32)
    targets = np.vstack([x for x in targets if x.size > 0]).astype(np.float32)

    # Save original length
    original_length = inputs.shape[0]

    if REMOVE_ALL_ZERO_SAMPS:
        # Create a boolean mask for rows where all values are zero in inputs or targets
        non_zero_mask = ~((inputs == 0).all(axis=1) | (targets == 0).all(axis=1))

        # Apply the mask to filter out rows
        inputs = inputs[non_zero_mask, :]  # Keep rows in inputs
        targets = targets[non_zero_mask, :]  # Keep rows in targets

    # Calculate the number of removed samples
    removed_samples = original_length - inputs.shape[0]

    print(f"Final input shape: {inputs.shape}")
    print(f"Final target shape: {targets.shape}")
    print(f"Samples included: {inputs.shape[0]}")
    print(f"Samples removed: {removed_samples}")

    return inputs, targets


# %%
# Load data 
BATCH_SIZE = 32

# Cache the data in .npy files for speed
def load_cached_data(input_file, target_file, input_dir, target_dir):
    if os.path.exists(input_file) and os.path.exists(target_file):
        inputs = np.load(input_file)
        targets = np.load(target_file)
    else:
        inputs, targets = load_dataset_parallel(input_dir, target_dir)
        np.save(input_file, inputs)  # Save inputs
        np.save(target_file, targets)  # Save targets
    
    return inputs, targets


# Common directory
def detect_base_directory():
    possible_dirs = [
        r"/home/muhammad.mahajna/workspace/research/data/cvr_est_project",  # Directory on TALC cluster
        r"/Users/muhammadmahajna/workspace/research/data/cvr_est_project"   # Directory on LAPTOP - abs path
    ] # Add your path here if the data is stored in a different location

    for base_dir in possible_dirs:
        if os.path.exists(base_dir):
            print(f"Using base directory: {base_dir}")
            return base_dir
    # Alternatively, you can use the host name, but it works just fine as is

    # Raise an error if no valid data directory is found
    raise ValueError("No valid base directory found.")

base_dir = detect_base_directory()

# Subdirectories - training/validation/testing
train_input_dir = os.path.join(base_dir, "func/registered/main_data/training")
val_input_dir = os.path.join(base_dir, "func/registered/main_data/validation")
test_input_dir = os.path.join(base_dir, "func/registered/main_data/testing")

train_target_dir = os.path.join(base_dir, "CVR_MAPS/registered/training")
val_target_dir = os.path.join(base_dir, "CVR_MAPS/registered/validation")
test_target_dir = os.path.join(base_dir, "CVR_MAPS/registered/testing")

# Load and preprocess the data
print("Loading training data...")
train_inputs, train_targets = load_cached_data("train_inputs.npy", "train_targets.npy", train_input_dir, train_target_dir)

print("Loading validation data...")
val_inputs, val_targets = load_cached_data("val_inputs.npy", "val_targets.npy", val_input_dir, val_target_dir)

print("Loading test data...")
test_inputs, test_targets = load_cached_data("test_inputs.npy", "test_targets.npy", test_input_dir, test_target_dir)

# Create datasets
train_dataset = TimeSeriesDataset(train_inputs, train_targets, normalize=NORMALIZE_DATA)
val_dataset = TimeSeriesDataset(val_inputs, val_targets, normalize=NORMALIZE_DATA)
test_dataset = TimeSeriesDataset(test_inputs, test_targets, normalize=NORMALIZE_DATA)

#Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %%
# Build model and training tools
INPUT_SIZE = 435 - REMOVE_TIME_POINT # Number of time points
LEARNING_RATE = 1e-3
model_cnn = CVRRegressionModel(input_size=INPUT_SIZE)
criterion = nn.MSELoss()
optimizer_cnn = torch.optim.Adam(model_cnn.parameters(), lr=LEARNING_RATE)

# %%
# Training function

# Parameters
EPOCHS = 10
MODEL_SAVE_PATH = "best_model_cnn.pth"
PATIENCE = 3  # Stop training if validation loss doesn't improve for 5 consecutive epochs
GRAD_CLIP = 1.0  # Gradient clipping max norm

def train_model(model, train_loader, val_loader, criterion, optimizer, device):
    model.to(device)
    best_val_loss = float("inf")  # Initialize with a large value
    epochs_without_improvement = 0  # Counter for epochs without improvement

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0  # Reset counter
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model saved with Validation Loss: {val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

        # Update the learning rate
        scheduler.step(val_loss)

        # Early stopping
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping triggered. No improvement for {PATIENCE} consecutive epochs.")
            break


# %%
# Train the model 
print("Training the model...")
train_model(model_cnn, train_loader, val_loader, criterion, optimizer_cnn, device)

# %%
# Evaluate on test data
if True:
    print("Evaluating the model on test data...")
    model_cnn.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model_cnn(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}")


