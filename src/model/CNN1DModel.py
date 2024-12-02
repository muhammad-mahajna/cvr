import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1DModel(nn.Module):
    def __init__(self, input_size, embedding_dim=40, max_len=5000):
        super(CNN1DModel, self).__init__()
        
        # Add positional encoding
        #self.positional_encoding = PositionalEncoding(embedding_dim, max_len)

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
        
        # Add Self-Attention after downsampling
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
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(160, 80),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(80, 1)
        )
    
    def forward(self, x):
        # Apply positional encoding
        
        #x = self.positional_encoding(x)
        
        # Pass through the initial block
        x = self.initial_block(x)
        
        # Pass through the residual block
        x = self.residual_block(x)
        
        # Pass through the self-attention block
        x, attention_weights = self.self_attention(x)
        
        # Continue with the middle block
        x = self.middle_block(x)
        
        # Pass through the multi-scale convolution block
        x = self.multi_scale_block(x)
        
        # Classifier layer
        x = self.classifier(x)
        return x, attention_weights

class MultiScaleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.fusion = nn.Conv1d(out_channels * 3, out_channels, kernel_size=1)  # fusion layer

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out = torch.cat([out1, out2, out3], dim=1)  # Concatenate along channel dimension
        return self.fusion(out)  # apply 1x1 convolution for feature fusion

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, embedding_dim=1, max_len=5000):
        """
        Positional Encoding for 1D sequences.

        Args:
            seq_len (int): Length of the sequence.
            embedding_dim (int): Channel dimension (should match input channels).
            max_len (int): Maximum sequence length for precomputing encodings.
        """
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim  # Typically matches the channel size
        self.seq_len = seq_len

        # Precompute positional encodings
        pe = torch.zeros(max_len, seq_len)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, seq_len, 2).float() * (-torch.log(torch.tensor(10000.0)) / seq_len))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as a buffer to prevent updates
        self.register_buffer('pe', pe.unsqueeze(0).transpose(1, 2))  # Shape: (1, seq_len, max_len)

    def forward(self, x):
        """
        Add positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, seq_len).

        Returns:
            torch.Tensor: Tensor with positional encodings added to the sequence dimension.
        """
        batch_size, channels, seq_len = x.size()

        # Ensure seq_len is within max_len
        assert seq_len <= self.pe.size(-1), "Sequence length exceeds the maximum length of positional encoding."

        # Add positional encoding to the sequence dimension
        return x + self.pe[:, :, :seq_len].expand(batch_size, channels, seq_len)
    

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _ = x.size()
        y = x.mean(dim=2)  # Global Average Pooling
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch, channels, 1)
        return x * y

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        # Learnable transformations for Query, Key, and Value
        self.query = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        # Scaling factor
        self.gamma = nn.Parameter(torch.zeros(1))  # Scales the output of the attention layer

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
