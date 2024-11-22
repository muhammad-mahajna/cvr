import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Parameters
INPUT_DIR = "/path/to/input/data"
TARGET_DIR = "/path/to/target/data"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading
def load_nifti(filepath):
    """Load a NIfTI file and return the data."""
    return nib.load(filepath).get_fdata()

def load_dataset(input_dir, target_dir):
    """Load input and target datasets from directories."""
    input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
    target_files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir)])
    
    inputs = [load_nifti(f) for f in input_files]
    targets = [load_nifti(f).reshape((-1, 1)) for f in target_files]
    
    inputs = np.vstack(inputs)
    targets = np.vstack(targets)
    
    return inputs, targets

# PyTorch Dataset
class FMRI_Dataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.targets[idx]
        return x.unsqueeze(1), y  # Add channel dimension for CNN

# Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.LeakyReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * (435 // 8), 1)  # Adjust based on input size and pooling

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = self.relu(self.pool(self.conv3(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Training
def train_model(model, train_loader, val_loader, criterion, optimizer, device):
    model.to(device)
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

# Main script
if __name__ == "__main__":
    print("Loading data...")
    inputs, targets = load_dataset(INPUT_DIR, TARGET_DIR)
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs, targets, test_size=0.1, random_state=42
    )
    
    train_dataset = FMRI_Dataset(inputs_train, targets_train)
    val_dataset = FMRI_Dataset(inputs_test, targets_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("Building model...")
    model = CNNModel()
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Training model...")
    train_model(model, train_loader, val_loader, criterion, optimizer, DEVICE)
    
    print("Saving model...")
    torch.save(model.state_dict(), "trained_model.pth")
