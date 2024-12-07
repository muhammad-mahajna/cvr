import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        """
        Attention mechanism for weighting the importance of time steps.
        Args:
        - hidden_size (int): Size of the hidden layer from LSTM.
        """
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output):
        """
        Forward pass for attention mechanism.
        Args:
        - lstm_output (Tensor): Output from LSTM, shape (batch_size, seq_length, hidden_size).
        Returns:
        - Tensor: Weighted sum of LSTM outputs, shape (batch_size, hidden_size).
        """
        # Calculate attention scores
        scores = self.attention_weights(lstm_output)  # Shape: (batch_size, seq_length, 1)
        scores = torch.softmax(scores, dim=1)  # Normalize scores
        # Apply weights to LSTM outputs
        context = torch.sum(scores * lstm_output, dim=1)  # Weighted sum across time steps
        return context


class LSTMRegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        """
        Bidirectional LSTM Model with Attention for regression.
        
        Args:
        - input_size (int): Number of input features per time step.
        - hidden_size (int): Number of hidden units in LSTM.
        - num_layers (int): Number of stacked LSTM layers.
        - dropout (float): Dropout rate between LSTM layers.
        """
        super(LSTMRegressionModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True  # Enable bidirectional LSTM
        )
        self.attention = Attention(hidden_size * 2)  # Bidirectional LSTM doubles hidden size
        self.fc = nn.Linear(hidden_size * 2, 1)  # Fully connected layer for regression output

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
        - x (Tensor): Input tensor of shape (batch_size, 1, sequence_length).
        Returns:
        - Tensor: Output tensor of shape (batch_size, 1).
        """
        # Transpose the input to match LSTM expectations
        x = x.transpose(1, 2)  # Shape: (batch_size, sequence_length, 1)
        lstm_out, _ = self.lstm(x)  # LSTM output, shape: (batch_size, sequence_length, hidden_size * 2)
        context = self.attention(lstm_out)  # Apply attention, shape: (batch_size, hidden_size * 2)
        out = self.fc(context)  # Fully connected layer, shape: (batch_size, 1)
        return out
