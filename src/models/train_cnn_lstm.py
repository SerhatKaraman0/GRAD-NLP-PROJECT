#!/usr/bin/env python
# filepath: /Users/user/Desktop/Projects/NLP-Learning/src/models/train_cnn_lstm.py

"""
Training script for the CNNBiLSTMClassifier model.

This script handles:
1. Data loading and preprocessing
2. Model instantiation
3. Training loop with validation
4. Model evaluation
5. Saving the trained model
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Import project modules
from src.core.logging_config import setup_logging, get_logger
from src.features.embeddings import EmbeddingProcessor
from src.models.models import CNNBiLSTMClassifier

# Setup logger
setup_logging()
logger = get_logger('CNNBiLSTM_Training')

class CNNBiLSTMTrainer:
    def __init__(self, config=None):
        """
        Initialize the trainer with configuration
        
        Args:
            config (dict): Configuration parameters
        """
        # Default configuration
        self.config = {
            'data_path': os.path.join(project_root, 'data', 'PREPROCESSED_Reviews.csv'),
            'models_dir': os.path.join(project_root, 'data', 'models'),
            'embedding_dim': 100,
            'max_features': 20000,
            'batch_size': 64,
            'epochs': 10,
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'max_seq_length': 500,
            'num_classes': 5,  # 1-5 star ratings
            'use_cuda': torch.cuda.is_available(),
            'random_state': 42,
            'early_stopping_patience': 3
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Device configuration
        self.device = torch.device('cuda' if self.config['use_cuda'] else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create directories if not exist
        os.makedirs(self.config['models_dir'], exist_ok=True)
        
    def load_data(self):
        """
        Load and preprocess the data
        
        Returns:
            tuple: Train and validation DataLoaders
        """
        logger.info("Loading data...")
        try:
            # Load preprocessed data
            df = pd.read_csv(self.config['data_path'])
            logger.info(f"Loaded {len(df)} records from {self.config['data_path']}")
            
            # Check required columns
            required_cols = ['Text', 'Score']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Initialize embedding processor
            embedding_processor = EmbeddingProcessor(
                batch_size=self.config['batch_size'],
                max_features=self.config['max_features'],
                embedding_dim=self.config['embedding_dim']
            )
            
            # Convert all Text values to strings to prevent errors with non-string values
            df['Text'] = df['Text'].astype(str)
            
            # Convert text to sequences and get pre-trained embedding matrix
            X_sequences, embedding_matrix = embedding_processor.prepare_embeddings(df['Text'].tolist())
            logger.info(f"Embedding matrix shape: {embedding_matrix.shape}")
            
            # Prepare target values (convert 1-5 scale to 0-4 for classification)
            y = df['Score'].values - 1
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_sequences, y, 
                test_size=self.config['validation_split'],
                random_state=self.config['random_state'],
                stratify=y
            )
            
            # Create DataLoaders
            train_dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.long),
                torch.tensor(y_train, dtype=torch.long)
            )
            
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.long),
                torch.tensor(y_val, dtype=torch.long)
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size']
            )
            
            logger.info(f"Data prepared: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
            return train_loader, val_loader, embedding_processor.max_features, embedding_matrix
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def train(self):
        """
        Train the CNNBiLSTMClassifier model
        
        Returns:
            tuple: Trained model and training history
        """
        try:
            # Load data
            train_loader, val_loader, vocab_size, embedding_matrix = self.load_data()
            
            # Initialize model
            model = CNNBiLSTMClassifier(
                input_size=self.config['max_seq_length'],
                embedding_dim=self.config['embedding_dim'],
                vocab_size=vocab_size + 1,  # +1 for padding token
                num_classes=self.config['num_classes']
            )
            
            # Initialize embedding layer with pre-trained weights
            logger.info(f"Initializing embedding layer with pre-trained GloVe {self.config['embedding_dim']}d embeddings")
            model.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            
            # Option to freeze the embedding layer
            freeze_embeddings = self.config.get('freeze_embeddings', False)
            if freeze_embeddings:
                logger.info("Freezing embedding layer - embeddings will not be updated during training")
                model.embedding.weight.requires_grad = False
            
            model = model.to(self.device)
            
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
            
            # Track training history
            history = {
                'train_loss': [],
                'val_loss': [],
                'train_accuracy': [],
                'val_accuracy': [],
                'best_val_loss': float('inf'),
                'best_epoch': 0
            }
            
            # Early stopping counter
            early_stop_counter = 0
            
            # Training loop
            logger.info(f"Starting training for {self.config['epochs']} epochs...")
            start_time = time.time()
            
            for epoch in range(self.config['epochs']):
                # Training phase
                model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward + backward + optimize
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    # Track statistics
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += targets.size(0)
                    train_correct += (predicted == targets).sum().item()
                    
                    # Log batch progress
                    if (batch_idx + 1) % 20 == 0:
                        logger.info(f'Epoch [{epoch+1}/{self.config["epochs"]}], '
                                    f'Batch [{batch_idx+1}/{len(train_loader)}], '
                                    f'Loss: {loss.item():.4f}')
                
                # Compute epoch training stats
                epoch_train_loss = train_loss / len(train_loader)
                epoch_train_acc = train_correct / train_total
                
                # Validation phase
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        
                        # Track statistics
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += targets.size(0)
                        val_correct += (predicted == targets).sum().item()
                
                # Compute epoch validation stats
                epoch_val_loss = val_loss / len(val_loader)
                epoch_val_acc = val_correct / val_total
                
                # Update history
                history['train_loss'].append(epoch_train_loss)
                history['val_loss'].append(epoch_val_loss)
                history['train_accuracy'].append(epoch_train_acc)
                history['val_accuracy'].append(epoch_val_acc)
                
                # Log epoch stats
                logger.info(f'Epoch [{epoch+1}/{self.config["epochs"]}], '
                           f'Train Loss: {epoch_train_loss:.4f}, '
                           f'Train Acc: {epoch_train_acc:.4f}, '
                           f'Val Loss: {epoch_val_loss:.4f}, '
                           f'Val Acc: {epoch_val_acc:.4f}')
                
                # Check for improvement
                if epoch_val_loss < history['best_val_loss']:
                    logger.info(f'Validation loss improved from {history["best_val_loss"]:.4f} to {epoch_val_loss:.4f}')
                    history['best_val_loss'] = epoch_val_loss
                    history['best_epoch'] = epoch + 1
                    
                    # Save best model
                    best_model_path = os.path.join(self.config['models_dir'], 'best_model_cnn_bilstm.pth')
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f'Best model saved to {best_model_path}')
                    
                    # Reset early stopping counter
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    logger.info(f'Validation loss did not improve. Counter: {early_stop_counter}/{self.config["early_stopping_patience"]}')
                    
                    # Check early stopping
                    if early_stop_counter >= self.config['early_stopping_patience']:
                        logger.info(f'Early stopping triggered after {epoch+1} epochs')
                        break
            
            # Training completed
            training_time = time.time() - start_time
            logger.info(f'Training completed in {training_time:.2f} seconds')
            
            # Save final model
            final_model_path = os.path.join(self.config['models_dir'], 'final_model_cnn_bilstm.pth')
            torch.save(model.state_dict(), final_model_path)
            logger.info(f'Final model saved to {final_model_path}')
            
            # Save training history
            history_path = os.path.join(self.config['models_dir'], 'training_history_cnn_bilstm.json')
            with open(history_path, 'w') as f:
                json.dump({k: v for k, v in history.items() if isinstance(v, list)}, f)
            logger.info(f'Training history saved to {history_path}')
            
            return model, history
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def evaluate(self, model, test_loader):
        """
        Evaluate the model on test data
        
        Args:
            model: The trained model
            test_loader: DataLoader for test data
            
        Returns:
            dict: Evaluation metrics
        """
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'rmse': rmse
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

def main():
    """Main function to train the model"""
    logger.info("Starting CNN BiLSTM model training")
    
    # Create trainer instance
    trainer = CNNBiLSTMTrainer()
    
    try:
        # Train model
        model, history = trainer.train()
        
        logger.info(f"Best validation loss: {history['best_val_loss']:.4f} at epoch {history['best_epoch']}")
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
