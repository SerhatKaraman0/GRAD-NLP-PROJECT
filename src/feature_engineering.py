from src.common_imports import *  # noqa: F403, F405
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
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import wget
import zipfile
import json
from gensim.models import Word2Vec

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineering(NlpModel):
    def __init__(self, batch_size=10_000, max_features=7000, embedding_dim=100):
        """Initialize paths and load dataset."""
        super().__init__()
        self.SAVE_DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.PREPROCESSED_DATA_PATH = os.path.join(self.SAVE_DATA_DIR, "PREPROCESSED_Reviews.csv")
        self.EMBEDDING_PATH = os.path.join(self.SAVE_DATA_DIR, "embeddings")
        self.batch_size = batch_size
        self.max_features = max_features
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.embedding_matrix = None
        os.makedirs(self.EMBEDDING_PATH, exist_ok=True)

    def prepare_embeddings(self, texts):
        """
        Prepare word embeddings using Word2Vec and GloVe.
        
        Args:
            texts: List of preprocessed text documents
            
        Returns:
            embedding_matrix: Pre-trained word embeddings
        """
        logger.info("Preparing word embeddings...")
        
        # Clean and tokenize texts
        cleaned_texts = []
        for text in texts:
            # Convert to string and clean
            text = str(text).lower()
            # Split into words and remove empty strings
            words = [word.strip() for word in text.split() if word.strip()]
            cleaned_texts.append(' '.join(words))
        
        # Tokenize texts
        logger.info("Tokenizing texts...")
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=self.max_features,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' ',
            oov_token='<UNK>'
        )
        self.tokenizer.fit_on_texts(cleaned_texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        
        # Get vocabulary info
        word_index = self.tokenizer.word_index
        logger.info(f"Found {len(word_index)} unique tokens")
        
        # Load pre-trained GloVe embeddings
        logger.info("Loading GloVe embeddings...")
        embeddings_index = {}
        glove_path = os.path.join(self.EMBEDDING_PATH, 'glove.6B.100d.txt')
        
        if not os.path.exists(glove_path):
            logger.info("Downloading GloVe embeddings...")
            # Download GloVe embeddings
            url = 'http://nlp.stanford.edu/data/glove.6B.zip'
            wget.download(url, self.EMBEDDING_PATH)
            with zipfile.ZipFile(os.path.join(self.EMBEDDING_PATH, 'glove.6B.zip'), 'r') as zip_ref:
                zip_ref.extractall(self.EMBEDDING_PATH)
        
        with open(glove_path, encoding='utf-8') as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, 'f', sep=' ')
                embeddings_index[word] = coefs
        
        logger.info(f"Loaded {len(embeddings_index)} word vectors from GloVe")
        
        # Create embedding matrix
        logger.info("Creating embedding matrix...")
        self.embedding_matrix = np.zeros((self.max_features, self.embedding_dim))
        for word, i in word_index.items():
            if i >= self.max_features:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector
        
        # Train Word2Vec on our corpus
        logger.info("Training Word2Vec on corpus...")
        tokenized_texts = [text.split() for text in cleaned_texts]
        word2vec_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.embedding_dim,
            window=5,
            min_count=1,
            workers=4
        )
        
        # Combine GloVe and Word2Vec embeddings
        logger.info("Combining GloVe and Word2Vec embeddings...")
        words_found = 0
        for word, i in word_index.items():
            if i >= self.max_features:
                continue
            if self.embedding_matrix[i].sum() == 0 and word in word2vec_model.wv:
                self.embedding_matrix[i] = word2vec_model.wv[word]
                words_found += 1
        
        logger.info(f"Added {words_found} word vectors from Word2Vec")
        
        return sequences, self.embedding_matrix

    def load_and_process_data(self):
        """Load and process the data including embedding preparation."""
        logger.info("Loading and processing data...")
        
        # Load preprocessed data
        df = pd.read_csv(self.PREPROCESSED_DATA_PATH)
        
        # Convert text to string and handle NaN values
        df['Text'] = df['Text'].fillna('').astype(str)
        
        # Map scores to sentiment (1-2: negative, 3: neutral, 4-5: positive)
        df['sentiment'] = df['Score'].apply(lambda x: 0 if x <= 2 else (2 if x >= 4 else 1))
        
        # Remove any rows with empty text
        df = df[df['Text'].str.strip() != '']
        
        # Use the 'Text' column for our analysis
        texts = df['Text'].values
        labels = df['sentiment'].values
        
        logger.info(f"Total samples: {len(texts)}")
        logger.info(f"Sample text: {texts[0][:100]}...")
        
        # Prepare embeddings
        sequences, embedding_matrix = self.prepare_embeddings(texts)
        
        # Pad sequences
        max_len = min(max(len(seq) for seq in sequences), 500)  # Cap at 500 tokens
        X = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Testing data shape: {X_test.shape}")
        logger.info(f"Label distribution: {np.bincount(labels)}")
        
        return X_train, y_train, X_test, y_test, embedding_matrix, max_len

    def build_lstm_model(self, input_size, embedding_dim=100, hidden_size=128, num_layers=2, dropout=0.5):
        """Build a PyTorch LSTM model instead of TensorFlow"""
        
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
        logger.info(f"Using device: {device}")
        model.to(device)
        
        # Track model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {trainable_params:,} trainable out of {total_params:,} total")
        
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
                    predicted = torch.round(outputs)
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
                        
                        # Calculate accuracy
                        predicted = torch.round(outputs)
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
                logger.info(f"Epoch {epoch+1}/{epochs} - "
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
                    logger.info(f"✅ New best model saved with val accuracy: {best_val_acc:.4f}")
                elif avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), os.path.join(self.SAVE_DATA_DIR, f'best_loss_model_{model_type}.pth'))
                    logger.info(f"✅ New best loss model saved: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement for {patience_counter} epochs")
                
                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"Early stopping after {epoch+1} epochs")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
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
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Calculate accuracy (rounded predictions)
        y_pred_rounded = np.round(y_pred)
        acc = accuracy_score(np.round(y_test), y_pred_rounded)
        
        # Log results
        logger.info(f"Model evaluation: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, Accuracy={acc:.4f}")
        
        # Create visualizations
        self._create_evaluation_visualizations(y_test, y_pred, mae)
        
        return mse, mae, acc, y_pred

    def _create_evaluation_visualizations(self, y_test, y_pred, mae):
        """Create and save evaluation visualizations."""
        metrics_dir = os.path.join(self.SAVE_DATA_DIR, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Create visualizations...
        # [Previous visualization code remains the same]

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
        
        logger.info(f"Training history plots saved to {metrics_dir}")


def main():
    """Main execution function."""
    # Check if PyTorch GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize FeatureEngineering class
    model = FeatureEngineering(batch_size=10_000, max_features=7000, embedding_dim=100)
    
    try:
        # Load data and prepare embeddings
        X_train, y_train, X_test, y_test, embedding_matrix, max_len = model.load_and_process_data()
        
        # Train and evaluate each model type separately
        model_types = ['ensemble']
        results = {}
        
        for model_type in model_types:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_type.upper()} LSTM Model")            
            logger.info(f"{'='*50}")
            
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
            model_path = os.path.join(model.SAVE_DATA_DIR, f"sentiment_model_{model_type}.pth")
            torch.save(trained_model.state_dict(), model_path)
            
            # Clear memory
            del trained_model
            torch.cuda.empty_cache()
            gc.collect()
        
        # Compare results
        results_df = pd.DataFrame(results).round(4)
        results_path = os.path.join(model.SAVE_DATA_DIR, 'model_comparison.csv')
        results_df.to_csv(results_path)
        logger.info(f"\nModel Comparison Results:\n{results_df}")
        
        logger.info("\nAll models have been trained and evaluated successfully!")
        
    except Exception as e:
        logger.error(f"Error in processing: {e}", exc_info=True)
        raise

    
if __name__ == "__main__":
    main()
