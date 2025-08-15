"""
Advanced neural network models for NLP tasks.
Contains improved architectures with state-of-the-art components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer-like architectures
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module
    """
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        
    def forward(self, x):
        # Input shape: [batch_size, seq_len, embed_dim]
        # Transpose to [seq_len, batch_size, embed_dim] for MultiheadAttention
        x = x.permute(1, 0, 2)
        attn_output, _ = self.multihead_attn(x, x, x)
        # Transpose back to [batch_size, seq_len, embed_dim]
        return attn_output.permute(1, 0, 2)

class GatedCNN(nn.Module):
    """
    Gated Convolutional Neural Network layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(GatedCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels * 2, kernel_size, padding=padding)
        self.gate_norm = nn.BatchNorm1d(out_channels)
        self.value_norm = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        # x shape: [batch_size, in_channels, seq_len]
        conved = self.conv(x)
        # Split channels for gating mechanism
        value, gate = torch.chunk(conved, 2, dim=1)
        # Apply batch normalization to value path
        value = self.value_norm(value)
        # Apply sigmoid to gate path (don't use GLU here as it reduces dimensions again)
        gate = self.gate_norm(gate)
        # Apply gating mechanism
        return value * torch.sigmoid(gate)

class AdvancedCNNBiLSTMClassifier(nn.Module):
    """
    Advanced CNN-BiLSTM classifier with self-attention and gated mechanisms
    """
    def __init__(self, input_size, embedding_dim=200, vocab_size=25000, num_classes=5, 
                 dropout_rate=0.5, spatial_dropout=0.4, lstm_hidden_size=256, lstm_layers=2):
        """
        Initialize Advanced CNN-BiLSTM classifier
        
        Args:
            input_size: Maximum sequence length
            embedding_dim: Dimension of word embeddings
            vocab_size: Size of vocabulary
            num_classes: Number of output classes
            dropout_rate: Dropout rate for fully connected layers
            spatial_dropout: Dropout rate for spatial dropout
            lstm_hidden_size: Hidden size for LSTM layers
            lstm_layers: Number of LSTM layers
        """
        super(AdvancedCNNBiLSTMClassifier, self).__init__()
        
        # Embedding layer - will be initialized with pre-trained weights
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Spatial dropout for better regularization
        self.spatial_dropout = nn.Dropout2d(spatial_dropout)
        
        # Gated CNN layers with different kernel sizes
        self.gated_conv3 = GatedCNN(embedding_dim, 128, kernel_size=3, padding=1)
        self.gated_conv5 = GatedCNN(embedding_dim, 128, kernel_size=5, padding=2)
        self.gated_conv7 = GatedCNN(embedding_dim, 128, kernel_size=7, padding=3)
        
        # Normalization and pooling
        self.batch_norm = nn.BatchNorm1d(384)  # 128*3 channels
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=384, 
            hidden_size=lstm_hidden_size, 
            num_layers=lstm_layers,
            batch_first=True, 
            bidirectional=True,
            dropout=0.3 if lstm_layers > 1 else 0
        )
        
        # Self-attention mechanism
        self.self_attention = MultiHeadSelfAttention(lstm_hidden_size*2, num_heads=8)
        
        # Layer normalization for attention outputs
        self.layer_norm = nn.LayerNorm(lstm_hidden_size*2)
        
        # Fully connected layers with residual connections
        self.fc_input_size = lstm_hidden_size * 2
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate * 0.8)
        self.dropout3 = nn.Dropout(dropout_rate * 0.6)
        
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Batch normalization for FC layers
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # Apply spatial dropout
        x = x.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
        x = self.spatial_dropout(x.unsqueeze(3)).squeeze(3)
        
        # Apply gated CNN layers
        x3 = self.gated_conv3(x)  # [batch_size, 128, seq_len]
        x5 = self.gated_conv5(x)  # [batch_size, 128, seq_len]
        x7 = self.gated_conv7(x)  # [batch_size, 128, seq_len]
        
        # Concatenate features from different convolutions
        x = torch.cat([x3, x5, x7], dim=1)  # [batch_size, 384, seq_len]
        
        # Apply batch normalization and max pooling
        x = self.batch_norm(x)
        x = self.maxpool(x)  # [batch_size, 384, seq_len/2]
        
        # Prepare for LSTM
        x = x.permute(0, 2, 1)  # [batch_size, seq_len/2, 384]
        
        # LSTM layers
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len/2, lstm_hidden_size*2]
        
        # Self-attention
        att_out = self.self_attention(lstm_out)
        
        # Residual connection and layer normalization
        x = self.layer_norm(lstm_out + att_out)
        
        # Global average pooling and max pooling
        avg_pool = torch.mean(x, 1)  # [batch_size, lstm_hidden_size*2]
        max_pool, _ = torch.max(x, 1)  # [batch_size, lstm_hidden_size*2]
        
        # Concatenate pooling results
        x = avg_pool + max_pool  # [batch_size, lstm_hidden_size*2]
        
        # Fully connected layers with residual connections
        x = self.dropout1(x)
        
        # First FC layer
        residual = x if self.fc_input_size == 256 else None
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        if residual is not None:
            x = x + residual
        
        # Second FC layer with residual connection
        residual = x
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Output layer
        x = self.dropout3(x)
        x = self.fc3(x)
        
        return x
