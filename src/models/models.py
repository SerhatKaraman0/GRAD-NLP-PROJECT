import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLSTMModel(nn.Module):
    def __init__(self, input_size, embedding_dim=100, hidden_size=128, num_layers=1, dropout=0.3):
        super(SimpleLSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        # Bidirectional LSTM has 2*hidden_size as output size
        self.fc = nn.Linear(hidden_size * 2, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        # Get the output for the last time step
        lstm_out = lstm_out[:, -1, :]
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out

class DeepLSTMModel(nn.Module):
    def __init__(self, input_size, embedding_dim=100, hidden_size=256, num_layers=2, dropout=0.5):
        super(DeepLSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        # Bidirectional LSTM has 2*hidden_size as output size
        self.fc = nn.Linear(hidden_size * 2, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        # Get the output for the last time step
        lstm_out = lstm_out[:, -1, :]
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out

class StackedLSTMModel(nn.Module):
    def __init__(self, input_size, embedding_dim=100):
        super(StackedLSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        lstm2_out, _ = self.lstm2(lstm1_out)
        # Get the output for the last time step
        lstm2_out = lstm2_out[:, -1, :]
        out = self.dropout2(lstm2_out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        
    def forward(self, x):
        # Get predictions from each model
        outputs = [model(x) for model in self.models]
        # Average the predictions
        return torch.mean(torch.stack(outputs), dim=0)

class CNNBiLSTMClassifier(nn.Module):
    def __init__(self, input_size, embedding_dim=100, vocab_size=20000, num_classes=5):
        """
        Initialize CNN-BiLSTM classifier with optional pre-trained embeddings
        
        Args:
            input_size: Maximum sequence length
            embedding_dim: Dimension of word embeddings (50, 100, or 200 for GloVe)
            vocab_size: Size of vocabulary
            num_classes: Number of output classes
        """
        super(CNNBiLSTMClassifier, self).__init__()
        
        # Embedding layer - will be initialized with pre-trained weights
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Improved spatial dropout for better regularization (increased from 0.3 to 0.4)
        self.spatial_dropout = nn.Dropout2d(0.4)
        
        # Multiple CNN layers with different kernel sizes for capturing different n-gram features
        self.conv1d_3 = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=3, padding=1)
        self.conv1d_5 = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=5, padding=2)
        self.conv1d_7 = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=7, padding=3)
        
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.batch_norm = nn.BatchNorm1d(192)  # 64*3 channels
        
        # Bidirectional LSTM with increased hidden size
        self.lstm = nn.LSTM(input_size=192, hidden_size=200, batch_first=True, bidirectional=True, dropout=0.3)
        
        # Attention mechanism
        self.attention = nn.Linear(400, 1)  # 200*2 due to bidirectional
        
        # Deeper fully connected layers with increased dropout for regularization
        self.dropout1 = nn.Dropout(0.6)  # Increased from 0.5 for better regularization
        self.dropout2 = nn.Dropout(0.4)  # Increased from 0.3 for better regularization
        self.fc1 = nn.Linear(400, 128)  # 200*2 from bidirectional LSTM
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        x = self.embedding(x)                       # [batch_size, seq_len, embedding_dim]
        
        # Simulate SpatialDropout1D: drop across embedding dim
        x = x.permute(0, 2, 1)                      # [batch_size, embedding_dim, seq_len]
        x = self.spatial_dropout(x.unsqueeze(3)).squeeze(3)
        
        # Apply multiple convolutions with different kernel sizes
        x_3 = F.relu(self.conv1d_3(x))              # [batch_size, 64, seq_len]
        x_5 = F.relu(self.conv1d_5(x))              # [batch_size, 64, seq_len]
        x_7 = F.relu(self.conv1d_7(x))              # [batch_size, 64, seq_len]
        
        # Concatenate the features from different convolutions
        x = torch.cat([x_3, x_5, x_7], dim=1)       # [batch_size, 192, seq_len]
        
        # Apply batch normalization
        x = self.batch_norm(x)
        
        # Max pooling and prepare for LSTM
        x = self.maxpool(x)                         # [batch_size, 192, seq_len/2]
        x = x.permute(0, 2, 1)                      # [batch_size, seq_len/2, 192]

        # LSTM layer
        lstm_out, _ = self.lstm(x)                  # [batch_size, seq_len/2, 400]
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)  # [batch_size, seq_len/2, 1]
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # [batch_size, 400]
        
        # Fully connected layers with residual connections
        out = self.dropout1(context_vector)
        out1 = F.relu(self.fc1(out))                # [batch_size, 128]
        out = self.dropout2(out1)
        out2 = F.relu(self.fc2(out))                # [batch_size, 64]
        out = self.fc3(out2)                        # [batch_size, num_classes]

        return out
