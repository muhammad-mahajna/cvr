
import torch.nn as nn

class CVRRegressionModel_base(nn.Module):
    def __init__(self, input_length=435, weight_decay=1e-4):
        super(CVRRegressionModel_base, self).__init__()
        self.model = nn.Sequential(
            # Conv Block 1
            nn.Conv1d(1, 20, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(20),
            nn.MaxPool1d(2),
            
            # Conv Block 2
            nn.Conv1d(20, 40, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(2),
            
            # Dropout after second block
            nn.Dropout(0.1),
            
            # Conv Block 3
            nn.Conv1d(40, 80, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(80),
            nn.MaxPool1d(2),
            
            # Conv Block 4
            nn.Conv1d(80, 160, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(160),
            nn.MaxPool1d(2),
            
            # Dropout after fourth block
            nn.Dropout(0.2),
            
            # Conv Block 5
            nn.Conv1d(160, 320, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(320),
            nn.MaxPool1d(2),
            
            # Flatten
            nn.Flatten(),
            
            # Dropout before fully connected layer
            nn.Dropout(0.2),
            
            # Fully connected regression output
            nn.Linear(self._calculate_flatten_size(input_length), 1)
        )
        
        # Weight decay regularizer
        self.weight_decay = weight_decay
    
    def forward(self, x):
        return self.model(x)
    
    def _calculate_flatten_size(self, input_length):
        """
        Calculate the output size after all Conv, Pooling, and Flatten layers.
        """
        size = input_length
        for _ in range(5):  # 5 MaxPooling layers with stride 2
            size = size // 2
        return size * 320  # Last Conv1D layer has 320 filters
