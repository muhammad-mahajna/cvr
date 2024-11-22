# %%
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

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')


# %%
# Data loaders

REMOVE_TIME_POINT = 5 # remove the first samples from the fMRI data

class TimeSeriesDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def load_dataset_parallel(input_dir, target_dir, num_workers=16):
    # List all files in the directories, sorted for consistency
    input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.nii', '.nii.gz'))])
    target_files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(('.nii', '.nii.gz'))])

    inputs = []
    targets = []

    # Use parallel processing to load inputs
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        inputs = list(executor.map(process_file, input_files, [False] * len(input_files), [REMOVE_TIME_POINT] * len(input_files)))
        targets = list(executor.map(process_file, target_files, [True] * len(target_files)))

    # Convert lists to NumPy arrays
    inputs = np.vstack(inputs).astype(np.float32)
    targets = np.vstack(targets).astype(np.float32)

    print(f"Final input shape: {inputs.shape}")
    print(f"Final target shape: {targets.shape}")

    return inputs, targets


# %%
# Load data 

def load_cached_data(input_file, target_file, input_dir, target_dir):
    if os.path.exists(input_file) and os.path.exists(target_file):
        inputs = np.load(input_file)
        targets = np.load(target_file)
    else:
        inputs, targets = load_dataset_parallel(input_dir, target_dir)
        np.save(input_file, inputs)  # Save inputs
        np.save(target_file, targets)  # Save targets
    
    return inputs, targets

BATCH_SIZE = 32

# Common directory
BASE_DIR = "/home/muhammad.mahajna/workspace/research/data/cvr_est_project"

# Subdirectories
TRAIN_INPUT_DIR = os.path.join(BASE_DIR, "func/registered/main_data/training")
VAL_INPUT_DIR = os.path.join(BASE_DIR, "func/registered/main_data/validation")
TEST_INPUT_DIR = os.path.join(BASE_DIR, "func/registered/main_data/testing")

TRAIN_TARGET_DIR = os.path.join(BASE_DIR, "CVR_MAPS/registered/training")
VAL_TARGET_DIR = os.path.join(BASE_DIR, "CVR_MAPS/registered/validation")
TEST_TARGET_DIR = os.path.join(BASE_DIR, "CVR_MAPS/registered/testing")

# Load and preprocess the data
print("Loading training data...")
train_inputs, train_targets = load_cached_data("train_inputs.npy", "train_targets.npy", TRAIN_INPUT_DIR, TRAIN_TARGET_DIR)

print("Loading validation data...")
val_inputs, val_targets = load_cached_data("val_inputs.npy", "val_targets.npy", VAL_INPUT_DIR, VAL_TARGET_DIR)

print("Loading test data...")
test_inputs, test_targets = load_cached_data("test_inputs.npy", "test_targets.npy", TEST_INPUT_DIR, TEST_TARGET_DIR)

# Create datasets and dataloaders
train_dataset = TimeSeriesDataset(train_inputs, train_targets)
val_dataset = TimeSeriesDataset(val_inputs, val_targets)
test_dataset = TimeSeriesDataset(test_inputs, test_targets)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# %%
# Model
class CNN1DModel(nn.Module):
    def __init__(self, input_size):
        super(CNN1DModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 20, kernel_size=3, padding=1),  # Conv1D Layer 1
            nn.LeakyReLU(),
            nn.BatchNorm1d(20),
            nn.MaxPool1d(2),

            nn.Conv1d(20, 40, kernel_size=3, padding=1),  # Conv1D Layer 2
            nn.LeakyReLU(),
            nn.MaxPool1d(2),

            nn.Dropout(0.1),  # Dropout for regularization

            nn.Conv1d(40, 80, kernel_size=3, padding=1),  # Conv1D Layer 3
            nn.LeakyReLU(),
            nn.BatchNorm1d(80),
            nn.MaxPool1d(2),

            nn.Conv1d(80, 160, kernel_size=3, padding=1),  # Conv1D Layer 4
            nn.LeakyReLU(),
            nn.BatchNorm1d(160),
            nn.MaxPool1d(2),

            nn.Dropout(0.2),  # Increased dropout for deeper layers

            nn.Conv1d(160, 320, kernel_size=3, padding=1),  # Conv1D Layer 5
            nn.LeakyReLU(),
            nn.BatchNorm1d(320),
            nn.MaxPool1d(2),

            nn.Flatten(),  # Flatten for fully connected layers
            nn.Dropout(0.2),
            nn.Linear(320 * (input_size // (2 ** 5)), 1)  # Final Dense Layer
        )

    def forward(self, x):
        return self.network(x)


# Build model
INPUT_SIZE = 435 - REMOVE_TIME_POINT # Number of time points
LEARNING_RATE = 1e-3
model = CNN1DModel(input_size=INPUT_SIZE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# %%
# Parameters
EPOCHS = 20
MODEL_SAVE_PATH = "best_model.pth"
PATIENCE = 7  # Stop training if validation loss doesn't improve for 5 consecutive epochs

def train_model(model, train_loader, val_loader, criterion, optimizer, device):
    model.to(device)
    best_val_loss = float("inf")  # Initialize with a large value
    epochs_without_improvement = 0  # Counter for epochs without improvement

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
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
                outputs = model(inputs)
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

        # Early stopping
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping triggered. No improvement for {PATIENCE} consecutive epochs.")
            break


# %%
# Train the model
print("Training the model...")
train_model(model, train_loader, val_loader, criterion, optimizer, device)


# %%
# Evaluate on test data
print("Evaluating the model on test data...")
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item() * inputs.size(0)
        print(loss.item())

test_loss /= len(test_loader.dataset)
print(f"Test Loss: {test_loss:.4f}")
