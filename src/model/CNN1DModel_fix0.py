# Model

import torch.nn as nn

class CNN1DModel(nn.Module):
    def __init__(self, input_size):
        super(CNN1DModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 20, kernel_size=3, padding=1),  # Conv1D Layer 1
            nn.BatchNorm1d(20),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(20, 40, kernel_size=3, padding=1),  # Conv1D Layer 2
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
            
            nn.Dropout(0.1),  # Dropout for regularization

            nn.Conv1d(40, 80, kernel_size=3, padding=1),  # Conv1D Layer 3
            nn.BatchNorm1d(80),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(80, 160, kernel_size=3, padding=1),  # Conv1D Layer 4
            nn.BatchNorm1d(160),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),

            nn.Dropout(0.2),  # Increased dropout for deeper layers

            nn.Conv1d(160, 320, kernel_size=3, padding=1),  # Conv1D Layer 5
            nn.BatchNorm1d(320),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),

            nn.Flatten(),  # Flatten for fully connected layers
            nn.Dropout(0.2),
            nn.Linear(320 * (input_size // (2 ** 5)), 1)  # Final Dense Layer
        )

    def forward(self, x):
        return self.network(x)
