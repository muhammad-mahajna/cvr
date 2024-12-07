import torch
import torch.nn as nn
import torch.nn.functional as F

# CVR regression model. A self contained class holding all model code. 
# Refer to the model_design.png for a visual representation of the model

class CVRRegressionModel(nn.Module):
    def __init__(self, embedding_dim=40):
        super(CVRRegressionModel, self).__init__()
        
        # Feature Extraction stage
        # Define the initial block
        self.initial_block = nn.Sequential(
            nn.Conv1d(1, 20, kernel_size=3, padding=1),
            nn.BatchNorm1d(20),
            nn.LeakyReLU(),
            SEBlock(20),  # SEBlock applied here
            nn.MaxPool1d(2)
        )
        
        # Define the residual block
        self.residual_block = nn.Sequential(
            ResidualBlock(20, embedding_dim),
            nn.MaxPool1d(2)
        )
        
        # Define Self-Attention block
        self.self_attention = SelfAttention(embedding_dim)
        
        # Define the middle block
        self.middle_block = nn.Sequential(
            nn.Conv1d(embedding_dim, 80, kernel_size=3, padding=1, dilation=2),
            nn.BatchNorm1d(80),
            nn.LeakyReLU(),
            SEBlock(80),  # SEBlock applied here as well
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        
        # Define the multi-scale convolution block
        self.multi_scale_block = nn.Sequential(
            MultiScaleConv1D(80, 160),
            nn.BatchNorm1d(160),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Regression stage
        # Define Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(160, 80),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(80, 1)
        )
    
    def forward(self, x):
        
        # Pass through the initial block
        x = self.initial_block(x)
        
        # Pass through the residual block
        x = self.residual_block(x)
        
        # Pass through the self-attention block
        x, attention_weights = self.self_attention(x)
        
        # Pass through the middle block
        x = self.middle_block(x)
        
        # Pass through the multi-scale convolution block
        x = self.multi_scale_block(x)
        
        # Finish with the regression layer
        x = self.classifier(x)
        return x, attention_weights

# Define the multi-scale feature extraction block
class MultiScaleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv1D, self).__init__()
        
        # Define the three convolution layers
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1) # conv_length = 3
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2) # conv_length = 5
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3) # conv_length = 7
        
        # Define the fusion conv layer
        self.fusion = nn.Conv1d(out_channels * 3, out_channels, kernel_size=1)      # fusion layer

    def forward(self, x):
        # Apply the three conv layers
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)

        # Concatenate along channel dimension
        out = torch.cat([out1, out2, out3], dim=1)
        
        # Apply 1x1 convolution for feature fusion
        return self.fusion(out) 

# Define the Squeeze and Excitation feature extraction block (SEBlock)
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        
        # Squeeze the input 
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.LeakyReLU()

        # Go back to the original dimention
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _ = x.size()
        # Global Average Pooling
        y = x.mean(dim=2)

        # Apply the FCCs
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch, channels, 1)

        # Multiply the input with the calculated mask
        return x * y

# Define the Residual feature extraction block (SEBlock)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        # Implement the standard resnet block (as in https://d2l.ai/chapter_convolutional-modern/resnet.html)

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # Calualte the skip function
        identity = self.skip(x)
        
        # Apply the main layers (2-convs followed by batch-norm and activation)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # The actual skipping
        out += identity

        # Final activation layer
        return self.relu(out)

# Define the Self Attention feature extraction block
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        # Learnable transformations for Query, Key, and Value
        self.query = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)

        # Scaling factor - Scales the output of the attention layer
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        # Compute Query, Key, and Value matrices
        query = self.query(x)  # Shape: (batch, channels//8, seq_len)
        key = self.key(x)      # Shape: (batch, channels//8, seq_len)
        value = self.value(x)  # Shape: (batch, channels, seq_len)

        # Compute attention scores
        scores = torch.bmm(query.permute(0, 2, 1), key)  # Shape: (batch, seq_len, seq_len)

        # Normalize scores with softmax
        attention_weights = F.softmax(scores, dim=-1)  # Shape: (batch, seq_len, seq_len)

        # Apply attention weights to the Value matrix
        out = torch.bmm(value, attention_weights)  # Shape: (batch, channels, seq_len)

        # Residual connection and scaling
        out = self.gamma * out + x

        return out, attention_weights

# EOF