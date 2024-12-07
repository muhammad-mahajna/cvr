# %%
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
from concurrent.futures import ProcessPoolExecutor
from utils import process_file
from CVRRegressionModel import CVRRegressionModel
from model.misc.LSTMRegressionModel import LSTMRegressionModel

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')


# %%

# Data loaders

REMOVE_TIME_POINT = 5  # Remove the first samples from the fMRI data
DATA_THRESHOLD = 1
ZERO_COUNT_THRESHOLD = 43 # 10%x430

class TimeSeriesDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def load_dataset_parallel(input_dir, target_dir, num_workers=1):
    # List all files in the directories, sorted for consistency
    input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.nii', '.nii.gz'))])
    target_files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(('.nii', '.nii.gz'))])

    inputs = []
    targets = []

    # Use parallel processing to load inputs and targets
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        inputs = list(executor.map(
            process_file, 
            input_files, 
            [False] * len(input_files), 
            [REMOVE_TIME_POINT] * len(input_files), 
            [DATA_THRESHOLD] * len(input_files), 
            [ZERO_COUNT_THRESHOLD] * len(input_files)
        ))
        targets = list(executor.map(process_file, target_files, [True] * len(target_files)))

    # Stack the loaded data
    inputs = np.vstack([x for x in inputs if x.size > 0]).astype(np.float32)
    targets = np.vstack([x for x in targets if x.size > 0]).astype(np.float32)

    # Save original length
    original_length = inputs.shape[0]

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

BASE_DIR = "/home/muhammad.mahajna/workspace/research/data/cvr_est_project"

# Subdirectories
TRAIN_INPUT_DIR = os.path.join(BASE_DIR, "func/registered/main_data/main_data/training")
VAL_INPUT_DIR = os.path.join(BASE_DIR, "func/registered/main_data/main_data/validation")
TEST_INPUT_DIR = os.path.join(BASE_DIR, "func/registered/main_data/main_data/testing")

TRAIN_TARGET_DIR = os.path.join(BASE_DIR, "CVR_MAPS/registered/registered/training")
VAL_TARGET_DIR = os.path.join(BASE_DIR, "CVR_MAPS/registered/registered/validation")
TEST_TARGET_DIR = os.path.join(BASE_DIR, "CVR_MAPS/registered/registered/testing")
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

# Build model
INPUT_SIZE = 435 - REMOVE_TIME_POINT # Number of time points
LEARNING_RATE = 1e-3
model_cnn = CVRRegressionModel()
criterion = nn.MSELoss()
optimizer_cnn = torch.optim.Adam(model_cnn.parameters(), lr=LEARNING_RATE)

model_lstm = LSTMRegressionModel(input_size=1, hidden_size=128, num_layers=2, dropout=0.3)
learning_rate = 0.001
optimizer_lstm = torch.optim.Adam(model_lstm.parameters(), lr=learning_rate)


# %%
# Parameters
EPOCHS = 10
MODEL_SAVE_PATH = "best_model_lstm.pth"
PATIENCE = 5  # Stop training if validation loss doesn't improve for 5 consecutive epochs
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
            outputs = model(inputs)
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

        # Update the learning rate
        scheduler.step(val_loss)

        # Early stopping
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping triggered. No improvement for {PATIENCE} consecutive epochs.")
            break


# %%
# Train the model - CNN
print("Training the model...")
#train_model(model_cnn, train_loader, val_loader, criterion, optimizer_cnn, device)

train_model(model_lstm, train_loader, val_loader, criterion, optimizer_lstm, device)


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

test_loss /= len(test_loader.dataset)
print(f"Test Loss: {test_loss:.4f}")


