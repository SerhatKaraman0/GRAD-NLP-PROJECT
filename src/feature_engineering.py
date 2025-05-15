from src.common_imports import * # noqa: F403, F405
from src.nlpmodel import NlpModel
from src.logging_config import *  # noqa: F403, F405
from utils.helper import CONTRACTIONS_DICT, SLANG_DICT  
from src.models import SimpleLSTMModel, DeepLSTMModel, StackedLSTMModel, EnsembleModel

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import gc
import sys
import logging
from tqdm import tqdm
import datetime

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    precision_score, recall_score, accuracy_score, 
    mean_squared_error, mean_absolute_error, r2_score
)
import io
import base64

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import swifter

from sklearn.feature_extraction.text import CountVectorizer

from scipy import sparse

import numpy as np
from tqdm import tqdm
import gc
import dask.dataframe as dd
import seaborn as sns


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

warnings.filterwarnings("ignore", category=FutureWarning)

console = Console()


class FeatureEngineering(NlpModel):
    def __init__(self):
            super().__init__()
            self.SAVE_DATA_DIR = os.path.join(self.BASE_DIR, "data")
            self.STATS_DIR = os.path.join(self.BASE_DIR, "stats")
            self.PREPROCESSED_DATA_DIR = os.path.join(self.SAVE_DATA_DIR, "PREPROCESSED_Reviews.csv")
            
            self.df = pd.read_csv(self.PREPROCESSED_DATA_DIR)
            self.df_size = len(self.df)

            self.batch_size = 10_000
            self.n_batches = (self.df_size + self.batch_size - 1) // self.batch_size

            self.vectorizer = CountVectorizer()

    def word_freq(self):
        X = self.vectorizer.fit_transform(self.df['Text'])
        
        word_counts = X.sum(axis=0).A1
        feature_names = self.vectorizer.get_feature_names_out() 
        
        freq = dict(zip(feature_names, word_counts))
        
        sorted_freq = sorted(freq.items(), key=lambda item: item[1], reverse=True)

        # Convert to DataFrame for plotting
        freq_df = pd.DataFrame(sorted_freq[:10], columns=['Word', 'Frequency'])

        sns.barplot(x='Word', y='Frequency', data=freq_df)
        plt.xticks(rotation=45)
        plt.show()
        
        

    def create_bow(self):
        """For creating the bag of words from preprocessed data and saving the result into a DataFrame"""
        self.self.logger.info("BAG OF WORDS ACTION HAS STARTED..")

        # Ensure NaN values are handled
        self.df['Text'] = self.df['Text'].fillna('')
        
        # Initialize CountVectorizer with more aggressive feature reduction
        self.vectorizer = CountVectorizer(
            min_df=5,            # Ignore terms that appear in less than 5 documents
            max_df=0.5,          # Ignore terms that appear in more than 50% of documents
            max_features=10000   # Only keep top 10,000 features
        )
        
        self.vectorizer.fit(self.df['Text'])

    def load_and_process_data(self):
        """Load and process the data including embedding preparation."""
        self.logger.info("Loading and processing data...")
        
        # Load preprocessed data
        df = pd.read_csv(self.PREPROCESSED_DATA_PATH)
        
        # Convert text to string and handle NaN values
        df['Text'] = df['Text'].fillna('').astype(str)
        
        # Use the original 1-5 score directly
        df['sentiment'] = df['Score']  # Use the actual 1-5 rating
        
        # Remove any rows with empty text
        df = df[df['Text'].str.strip() != '']
        
        # Use the 'Text' column for our analysis
        texts = df['Text'].values
        labels = df['sentiment'].values
        
        self.logger.info(f"Total samples: {len(texts)}")
        self.logger.info(f"Sample text: {texts[0][:100]}...")
        
        # Prepare embeddings
        sequences, embedding_matrix = self.prepare_embeddings(texts)
        
        # Pad sequences
        max_len = min(max(len(seq) for seq in sequences), 500)  # Cap at 500 tokens
        X = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        self.logger.info(f"Training data shape: {X_train.shape}")
        self.logger.info(f"Testing data shape: {X_test.shape}")
        self.logger.info(f"Score distribution: {np.bincount(labels.astype(int))}")
        
        return X_train, y_train, X_test, y_test, embedding_matrix, max_len

        # Process in smaller batches to reduce memory usage
        batch_size = min(5000, self.batch_size)  # Use smaller batches if needed
        
        # Define a PyTorch LSTM model
        class LSTMModel(nn.Module):
            def __init__(self, input_size, embedding_dim, hidden_size, num_layers, dropout):
                super(LSTMModel, self).__init__()
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
                
        return LSTMModel(input_size, embedding_dim, hidden_size, num_layers, dropout)
    
    def train_model(self, X, y, embedding_matrix, max_len, model_type='ensemble'):
        """Train the PyTorch model with improved progress tracking."""
        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Use a smaller batch size if running out of memory
        batch_size = 128 if torch.cuda.is_available() else 64
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Build model based on type
        if model_type == 'simple':
            model = self.build_lstm_model(
                input_size=len(embedding_matrix), 
                embedding_dim=100, 
                hidden_size=128, 
                num_layers=1, 
                dropout=0.3
            )
        elif model_type == 'deep':
            model = self.build_lstm_model(
                input_size=len(embedding_matrix), 
                embedding_dim=100, 
                hidden_size=256, 
                num_layers=2, 
                dropout=0.5
            )
        elif model_type == 'stacked':
            model = self.build_stacked_lstm_model(
                input_size=len(embedding_matrix), 
                embedding_dim=100
            )
        elif model_type == 'ensemble':
            # Create ensemble of models
            models = []
            for m_type in ['simple', 'deep', 'stacked']:
                if m_type == 'stacked':
                    models.append(self.build_stacked_lstm_model(
                        input_size=len(embedding_matrix), 
                        embedding_dim=100
                    ))
                else:
                    hidden_size = 128 if m_type == 'simple' else 256
                    num_layers = 1 if m_type == 'simple' else 2
                    models.append(self.build_lstm_model(
                        input_size=len(embedding_matrix), 
                        embedding_dim=100, 
                        hidden_size=hidden_size, 
                        num_layers=num_layers, 
                        dropout=0.5
                    ))
            # Use the first model for now
            model = models[0]
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Move model to GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {device}")
        model.to(device)
        
        # Track model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: {trainable_params:,} trainable out of {total_params:,} total")
        
        # Set up optimizer and loss function with learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        criterion = nn.MSELoss()
        
        # Training loop
        epochs = 20
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience = 7
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
        
        # Create progress bar for epochs
        epoch_bar = tqdm(range(epochs), desc=f"Training {model_type} model", position=0)
        
        try:
            for epoch in epoch_bar:
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                # Training phase
                batch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                                leave=False, position=1)
                for batch_X, batch_y in batch_bar:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    # Forward pass
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # Calculate accuracy (rounded predictions)
                    # For 1-5 scores, we need to clamp and round
                    predicted = torch.round(torch.clamp(outputs, 1, 5))
                    train_total += batch_y.size(0)
                    train_correct += (predicted == batch_y).sum().item()
                    current_acc = train_correct / max(1, train_total)
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                    # Update batch progress bar
                    batch_bar.set_postfix({
                        'loss': f"{loss.item():.4f}", 
                        'acc': f"{current_acc:.4f}"
                    })
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", 
                                  leave=False, position=1)
                    for batch_X, batch_y in val_bar:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        
                        # Calculate accuracy with clamping for 1-5 range
                        predicted = torch.round(torch.clamp(outputs, 1, 5))
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
                        current_val_acc = val_correct / max(1, val_total)
                        
                        val_loss += loss.item()
                        
                        # Update validation bar
                        val_bar.set_postfix({
                            'loss': f"{loss.item():.4f}", 
                            'acc': f"{current_val_acc:.4f}"
                        })
                
                # Calculate average losses and accuracies
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                train_accuracy = train_correct / max(1, train_total)
                val_accuracy = val_correct / max(1, val_total)
                
                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                
                # Update history
                history['train_loss'].append(avg_train_loss)
                history['val_loss'].append(avg_val_loss)
                history['train_acc'].append(train_accuracy)
                history['val_acc'].append(val_accuracy)
                history['lr'].append(current_lr)
                
                # Update scheduler
                scheduler.step(avg_val_loss)
                
                # Update epoch progress bar
                epoch_bar.set_postfix({
                    'train_loss': f"{avg_train_loss:.4f}",
                    'val_loss': f"{avg_val_loss:.4f}", 
                    'train_acc': f"{train_accuracy:.4f}", 
                    'val_acc': f"{val_accuracy:.4f}",
                    'lr': f"{current_lr:.6f}"
                })
                
                # Print statistics
                self.logger.info(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, "
                      f"Train Acc: {train_accuracy:.4f}, "
                      f"Val Acc: {val_accuracy:.4f}, "
                      f"LR: {current_lr:.6f}")
                
                # Save the best model
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    patience_counter = 0
                    torch.save(model.state_dict(), os.path.join(self.SAVE_DATA_DIR, f'best_model_{model_type}.pth'))
                    self.logger.info(f"✅ New best model saved with val accuracy: {best_val_acc:.4f}")
                elif avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), os.path.join(self.SAVE_DATA_DIR, f'best_loss_model_{model_type}.pth'))
                    self.logger.info(f"✅ New best loss model saved: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    self.logger.info(f"No improvement for {patience_counter} epochs")
                
                # Early stopping
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping after {epoch+1} epochs")
                    break
                    
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        
        # Save training history
        self.save_training_history(history, model_type)
        
        # Plot training history
        self._plot_training_history(history, model_type)
        
        return model, history
    
    def build_stacked_lstm_model(self, input_size, embedding_dim=100):
        """Build a PyTorch stacked LSTM model"""
        class StackedLSTMModel(nn.Module):
            def __init__(self, input_size, embedding_dim):
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
                
        return StackedLSTMModel(input_size, embedding_dim)
    
    def save_training_history(self, history, model_type):
        """Save training history to file."""
        history_path = os.path.join(self.SAVE_DATA_DIR, f'training_history_{model_type}.json')
        with open(history_path, 'w') as f:
            # Convert tensors/numpy arrays to Python lists for JSON serialization
            serializable_history = {}
            for key, values in history.items():
                serializable_history[key] = [float(val) for val in values]
            json.dump(serializable_history, f)
    
    def evaluate_model(self, model, X_test, y_test, model_type='ensemble'):
        """Evaluate the PyTorch model on test data."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        # Convert to PyTorch tensors
        X_test_tensor = torch.tensor(X_test, dtype=torch.long).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
        
        # Create dataloader for batched evaluation
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor.view(-1, 1))
        test_loader = DataLoader(test_dataset, batch_size=256)
        
        # Evaluate
        test_loss = 0.0
        all_preds = []
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for batch_X, batch_y in tqdm(test_loader, desc="Evaluating"):
                outputs = model(batch_X)
                test_loss += criterion(outputs, batch_y).item()
                all_preds.append(outputs.cpu().numpy())
        
        # Combine predictions and convert to numpy
        y_pred = np.vstack(all_preds).flatten()
        
        # Clamp predictions to valid score range (1-5)
        y_pred_clamped = np.clip(y_pred, 1, 5)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred_clamped)
        mse = mean_squared_error(y_test, y_pred_clamped)
        rmse = np.sqrt(mse)
        
        # Calculate accuracy (rounded predictions within 1-5 range)
        y_pred_rounded = np.round(y_pred_clamped)
        acc = accuracy_score(y_test, y_pred_rounded)
        
        # Log results
        self.logger.info(f"Model evaluation: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, Accuracy={acc:.4f}")
        
        # Create visualizations
        self._create_evaluation_visualizations(y_test, y_pred_clamped, mae)
        
        return mse, mae, acc, y_pred_clamped

    def _create_evaluation_visualizations(self, y_test, y_pred, mae):
        """Create and save evaluation visualizations."""
        metrics_dir = os.path.join(self.SAVE_DATA_DIR, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Create a confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(np.round(y_test), np.round(y_pred))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(metrics_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Create a distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(y_test, alpha=0.5, label='True Values')
        plt.hist(y_pred, alpha=0.5, label='Predictions')
        plt.title('Distribution of True vs Predicted Values')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(metrics_dir, 'distribution_plot.png'))
        plt.close()
        
        # Create a scatter plot of true vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.plot([1, 5], [1, 5], 'r--')  # Diagonal line for perfect predictions
        plt.title(f'True vs Predicted Values (MAE: {mae:.4f})')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.grid(True)
        plt.savefig(os.path.join(metrics_dir, 'true_vs_pred.png'))
        plt.close()

    def _plot_training_history(self, history, model_type):
        """Create and save plots of training history."""
        metrics_dir = os.path.join(self.SAVE_DATA_DIR, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Create a figure with subplots
        plt.figure(figsize=(15, 10))
        
        # Plot training & validation loss
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Training')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        
        # Plot training & validation accuracy
        plt.subplot(2, 2, 2)
        plt.plot(history['train_acc'], label='Training')
        plt.plot(history['val_acc'], label='Validation')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate
        plt.subplot(2, 2, 3)
        plt.plot(history['lr'])
        plt.title('Learning Rate')
        plt.ylabel('Learning Rate')
        plt.xlabel('Epoch')
        plt.grid(True)
        
        # Plot loss vs accuracy
        plt.subplot(2, 2, 4)
        plt.scatter(history['train_loss'], history['train_acc'], label='Training')
        plt.scatter(history['val_loss'], history['val_acc'], label='Validation')
        plt.title('Loss vs. Accuracy')
        plt.xlabel('Loss')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, f'training_history_{model_type}.png'))
        plt.close()
        
        self.logger.info(f"Training history plots saved to {metrics_dir}")

    def generate_model_dashboard(self, model_results, model_type='ensemble'):
        """
        Generate an HTML dashboard for model metrics and visualizations.
        
        Args:
            model_results: Dictionary containing model evaluation metrics
            model_type: Type of model ('simple', 'deep', 'stacked', or 'ensemble')
        """
        metrics_dir = os.path.join(self.SAVE_DATA_DIR, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Extract metrics
        mse = model_results.get('mse', 0)
        mae = model_results.get('mae', 0)
        accuracy = model_results.get('accuracy', 0)
        precision = model_results.get('precision', 0)
        recall = model_results.get('recall', 0)
        f1 = model_results.get('f1', 0)
        rmse = np.sqrt(mse) if mse else 0
        r2 = model_results.get('r2', float('nan'))
        
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Convert images to base64 for embedding in HTML
        def image_to_base64(image_path):
            if os.path.exists(image_path):
                with open(image_path, "rb") as img_file:
                    return base64.b64encode(img_file.read()).decode('utf-8')
            return ""
        
        confusion_matrix_b64 = image_to_base64(os.path.join(metrics_dir, 'confusion_matrix.png'))
        true_vs_pred_b64 = image_to_base64(os.path.join(metrics_dir, 'true_vs_pred.png'))
        distribution_plot_b64 = image_to_base64(os.path.join(metrics_dir, 'distribution_plot.png'))
        training_history_b64 = image_to_base64(os.path.join(metrics_dir, f'training_history_{model_type}.png'))
        
        # HTML template
        html_content = f'''<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Sentiment Analysis Model Dashboard</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        margin: 0;
                        padding: 20px;
                        color: #333;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                    }}
                    .header {{
                        background-color: #4a86e8;
                        color: white;
                        padding: 20px;
                        text-align: center;
                        border-radius: 5px;
                        margin-bottom: 20px;
                    }}
                    .metric-box {{
                        background-color: #f9f9f9;
                        border-radius: 5px;
                        padding: 15px;
                        margin-bottom: 15px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .metrics-container {{
                        display: flex;
                        flex-wrap: wrap;
                        justify-content: space-between;
                        margin-bottom: 20px;
                    }}
                    .metric-item {{
                        width: 22%;
                        text-align: center;
                        background-color: #e8f4f8;
                        padding: 15px;
                        border-radius: 5px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .metric-value {{
                        font-size: 24px;
                        font-weight: bold;
                        color: #4a86e8;
                    }}
                    .charts-container {{
                        display: flex;
                        flex-wrap: wrap;
                        justify-content: space-between;
                    }}
                    .chart-box {{
                        width: 48%;
                        margin-bottom: 20px;
                        background-color: #f9f9f9;
                        border-radius: 5px;
                        padding: 15px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .chart-box img {{
                        width: 100%;
                        height: auto;
                    }}
                    .full-width {{
                        width: 100%;
                    }}
                    h2 {{
                        color: #4a86e8;
                    }}
                    @media (max-width: 768px) {{
                        .metric-item {{
                            width: 48%;
                            margin-bottom: 15px;
                        }}
                        .chart-box {{
                            width: 100%;
                        }}
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Sentiment Analysis Model Dashboard</h1>
                        <p>Model performance metrics and visualizations for {model_type.upper()} model</p>
                    </div>
                    
                    <div class="metric-box">
                        <h2>Key Performance Metrics</h2>
                        <div class="metrics-container">
                            <div class="metric-item">
                                <h3>Accuracy</h3>
                                <div class="metric-value">{accuracy:.4f}</div>
                                <p>Classification Accuracy</p>
                            </div>
                            <div class="metric-item">
                                <h3>MAE</h3>
                                <div class="metric-value">{mae:.4f}</div>
                                <p>Mean Absolute Error</p>
                            </div>
                            <div class="metric-item">
                                <h3>MSE</h3>
                                <div class="metric-value">{mse:.4f}</div>
                                <p>Mean Squared Error</p>
                            </div>
                            <div class="metric-item">
                                <h3>RMSE</h3>
                                <div class="metric-value">{rmse:.4f}</div>
                                <p>Root Mean Squared Error</p>
                            </div>
                            <div class="metric-item">
                                <h3>R²</h3>
                                <div class="metric-value">{r2}</div>
                                <p>Coefficient of Determination</p>
                            </div>
                            <div class="metric-item">
                                <h3>Precision</h3>
                                <div class="metric-value">{precision:.4f}</div>
                                <p>Precision Score</p>
                            </div>
                            <div class="metric-item">
                                <h3>Recall</h3>
                                <div class="metric-value">{recall:.4f}</div>
                                <p>Recall Score</p>
                            </div>
                            <div class="metric-item">
                                <h3>F1</h3>
                                <div class="metric-value">{f1:.4f}</div>
                                <p>F1 Score</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="charts-container">
                        <div class="chart-box">
                            <h2>Actual vs Predicted Ratings</h2>
                            <img src="data:image/png;base64,{true_vs_pred_b64}">
                        </div>
                        <div class="chart-box">
                            <h2>Error Distribution</h2>
                            <img src="data:image/png;base64,{distribution_plot_b64}">
                        </div>
                        <div class="chart-box full-width">
                            <h2>Confusion Matrix (Rounded Ratings)</h2>
                            <img src="data:image/png;base64,{confusion_matrix_b64}">
                        </div>
                        <div class="chart-box full-width">
                            <h2>Training History</h2>
                            <img src="data:image/png;base64,{training_history_b64}">
                        </div>
                    </div>
                    
                    <div class="metric-box">
                        <h2>Model Information</h2>
                        <p><strong>Features:</strong> {self.max_features} embedding features</p>
                        <p><strong>Architecture:</strong> {model_type.upper()} LSTM model</p>
                        <p><strong>Embedding Dimension:</strong> {self.embedding_dim}</p>
                        <p><strong>Generated:</strong> {timestamp}</p>
                    </div>
                </div>
            </body>
            </html>'''
        
        # Write HTML to file
        dashboard_path = os.path.join(metrics_dir, f'model_dashboard_{model_type}.html')
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Model dashboard generated at {dashboard_path}")
        
        # Generate a combined dashboard for all models if this is the last model
        if model_type == 'ensemble':
            self.generate_combined_dashboard()
        
        return dashboard_path

    def generate_combined_dashboard(self):
        """Generate a combined dashboard comparing all model types"""
        metrics_dir = os.path.join(self.SAVE_DATA_DIR, "metrics")
        
        # Read comparison results
        comparison_path = os.path.join(self.SAVE_DATA_DIR, 'model_comparison.csv')
        if not os.path.exists(comparison_path):
            self.logger.warning(f"Model comparison file not found at {comparison_path}")
            return
        
        # Load comparison data
        try:
            comparison_df = pd.read_csv(comparison_path)
            # Convert DataFrame to HTML table
            comparison_table = comparison_df.to_html(classes='comparison-table', border=0)
        except Exception as e:
            self.logger.error(f"Error loading comparison data: {e}")
            comparison_table = "<p>Error loading comparison data</p>"
        
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # HTML for combined dashboard
        html_content = f'''<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Combined Model Comparison Dashboard</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        margin: 0;
                        padding: 20px;
                        color: #333;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                    }}
                    .header {{
                        background-color: #4a86e8;
                        color: white;
                        padding: 20px;
                        text-align: center;
                        border-radius: 5px;
                        margin-bottom: 20px;
                    }}
                    .comparison-box {{
                        background-color: #f9f9f9;
                        border-radius: 5px;
                        padding: 15px;
                        margin-bottom: 15px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        overflow-x: auto;
                    }}
                    .model-links {{
                        display: flex;
                        justify-content: space-around;
                        margin: 20px 0;
                    }}
                    .model-link {{
                        display: inline-block;
                        padding: 10px 15px;
                        background-color: #4a86e8;
                        color: white;
                        text-decoration: none;
                        border-radius: 5px;
                        font-weight: bold;
                    }}
                    .comparison-table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 15px 0;
                    }}
                    .comparison-table th, .comparison-table td {{
                        padding: 10px;
                        text-align: center;
                        border-bottom: 1px solid #ddd;
                    }}
                    .comparison-table th {{
                        background-color: #e8f4f8;
                        color: #333;
                    }}
                    .comparison-table tr:hover {{
                        background-color: #f5f5f5;
                    }}
                    .footer {{
                        margin-top: 20px;
                        text-align: center;
                        color: #666;
                        font-size: 0.9em;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Sentiment Analysis Model Comparison</h1>
                        <p>Comparative analysis of all trained models</p>
                    </div>
                    
                    <div class="comparison-box">
                        <h2>Model Performance Comparison</h2>
                        {comparison_table}
                    </div>
                    
                    <div class="comparison-box">
                        <h2>Individual Model Dashboards</h2>
                        <p>Click on a model type to view its detailed dashboard:</p>
                        <div class="model-links">
                            <a href="model_dashboard_simple.html" class="model-link">Simple LSTM</a>
                            <a href="model_dashboard_deep.html" class="model-link">Deep LSTM</a>
                            <a href="model_dashboard_stacked.html" class="model-link">Stacked LSTM</a>
                            <a href="model_dashboard_ensemble.html" class="model-link">Ensemble</a>
                        </div>
                    </div>
                    
                    <div class="footer">
                        <p>Generated on: {timestamp}</p>
                    </div>
                </div>
            </body>
            </html>'''
        
        # Write HTML to file
        dashboard_path = os.path.join(metrics_dir, 'model_dashboard.html')
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Combined model dashboard generated at {dashboard_path}")
        
        return dashboard_path
def main():
    """Main execution function."""
    # Check if PyTorch GPU is available
    
    # Initialize FeatureEngineering class
    model = FeatureEngineering(batch_size=10_000, max_features=7000, embedding_dim=100)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        model.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        # Load data and prepare embeddings
        X_train, y_train, X_test, y_test, embedding_matrix, max_len = model.load_and_process_data()
        
        # Train and evaluate each model type separately
        model_types = ['deep', 'stacked', 'ensemble']
        results = {}
        
        for model_type in model_types:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Training {model_type.upper()} LSTM Model")            
            self.logger.info(f"{'='*50}")
            
            # Train model with PyTorch
            trained_model, history = model.train_model(
                X_train, y_train, 
                embedding_matrix, max_len,
                model_type=model_type
            )
            
            # Evaluate model
            mse, mae, acc, y_pred = model.evaluate_model(
                trained_model, X_test, y_test, 
                model_type=model_type
            )
            
            # Store results
            results[model_type] = {
                'mse': mse,
                'mae': mae,
                'accuracy': acc,
                'precision': precision_score(np.round(y_test), np.round(y_pred), zero_division=0),
                'recall': recall_score(np.round(y_test), np.round(y_pred), zero_division=0),
                'f1': f1_score(np.round(y_test), np.round(y_pred), zero_division=0)
            }
            
            # Save model and results
            model_path = os.path.join(model.SAVE_DATA_DIR, f"sentiment_model_v2{model_type}.pth")
            torch.save(trained_model.state_dict(), model_path)
            
            model.generate_model_dashboard(results[model_type], model_type=model_type)
                  # Clear memory
            del trained_model
            torch.cuda.empty_cache()
            gc.collect()

        sparse_matrices = sparse.vstack(sparse_matrices)
        feature_names = self.vectorizer.get_feature_names_out()
        bow_df = pd.DataFrame.sparse.from_spmatrix(sparse_matrices, columns=feature_names)

        self.bow_df = bow_df
        return bow_df

    def save_to_parquet(self, df, output_path: str = "processed_data.parquet") -> None:
        """Save the processed DataFrame to Parquet with gzip compression"""
        self.self.logger.info("SAVING TO PARQUET STARTED..")
        self.print_section("SAVING TO PARQUET STARTED..")

        # Check if DataFrame has sparse data
        has_sparse = hasattr(df, 'sparse') and hasattr(df.sparse, 'to_dense')
    
        if has_sparse:
            # Convert sparse DataFrame to dense
            dense_df = df.sparse.to_dense()
        else:
            self.self.logger.warning("Input DataFrame does not contain sparse data.")
            dense_df = df
    
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
        # Save to Parquet format with gzip compression
        dense_df.to_parquet(f"{output_path}.parquet.gz", compression="gzip")

        self.self.logger.info(f"DF SAVED TO {output_path}.parquet.gz")

        return output_path


       


if __name__ == "__main__":
    main()
