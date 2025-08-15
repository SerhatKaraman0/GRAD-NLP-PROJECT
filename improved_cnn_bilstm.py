#!/usr/bin/env python
# filepath: /Users/user/Desktop/Projects/NLP-Learning/improved_cnn_bilstm.py

"""
Improved CNN BiLSTM training script with:
1. Adjusted regularization parameters
2. Advanced learning rate scheduling
5. K-fold cross-validation
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
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

# Add project root to sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import project modules
from src.core.logging_config import setup_logging, get_logger
from src.features.embeddings import EmbeddingProcessor
from src.models.models import CNNBiLSTMClassifier

# Setup logger
setup_logging()
logger = get_logger('Improved_CNNBiLSTM_Training')

class ImprovedCNNBiLSTMTrainer:
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
            'epochs': 15,  # Increased epochs for better training
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'max_seq_length': 500,
            'num_classes': 5,  # 1-5 star ratings
            'use_cuda': torch.cuda.is_available(),
            'random_state': 42,
            'early_stopping_patience': 5,  # Increased patience
            'k_folds': 3,  # Number of folds for cross-validation
            'weight_decay': 0.02,  # Increased from 0.01
            'use_kfold': True  # Whether to use k-fold cross-validation
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
            tuple: Processed data, embedding matrix, vocab size, and sequence length
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
            
            # Pad sequences to the same length
            max_len = min(self.config['max_seq_length'], max(len(seq) for seq in X_sequences))
            logger.info(f"Padding sequences to length: {max_len}")
            
            # Create a manual padding function using numpy
            def pad_sequences(sequences, max_len, padding='post'):
                padded_seqs = []
                for seq in sequences:
                    if len(seq) > max_len:
                        padded_seq = seq[:max_len]
                    else:
                        if padding == 'post':
                            padded_seq = np.concatenate([seq, np.zeros(max_len - len(seq), dtype=int)])
                        else:  # pre padding
                            padded_seq = np.concatenate([np.zeros(max_len - len(seq), dtype=int), seq])
                    padded_seqs.append(padded_seq)
                return np.array(padded_seqs)
                
            X_padded = pad_sequences(X_sequences, max_len, padding='post')
            
            return X_padded, y, embedding_matrix, embedding_processor.max_features, max_len
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_data_loaders(self, X_train, y_train, X_val, y_val):
        """Create PyTorch DataLoaders from training and validation data"""
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
        
        return train_loader, val_loader
    
    def init_model(self, vocab_size, embedding_matrix, max_len):
        """Initialize and prepare the model"""
        # Initialize model
        model = CNNBiLSTMClassifier(
            input_size=max_len,  # Use the actual padded sequence length
            embedding_dim=self.config['embedding_dim'],
            vocab_size=vocab_size + 1,  # +1 for padding token
            num_classes=self.config['num_classes']
        )
        
        # Initialize embedding layer with pre-trained weights
        logger.info(f"Initializing embedding layer with pre-trained GloVe {self.config['embedding_dim']}d embeddings")
        
        # Add a row of zeros at the beginning for the padding token (index 0)
        padded_embedding_matrix = np.zeros((vocab_size + 1, self.config['embedding_dim']))
        padded_embedding_matrix[1:] = embedding_matrix[:vocab_size]
        
        # Copy the padded embedding matrix to the model's embedding layer
        model.embedding.weight.data.copy_(torch.from_numpy(padded_embedding_matrix))
        
        # Option to freeze the embedding layer
        freeze_embeddings = self.config.get('freeze_embeddings', False)
        if freeze_embeddings:
            logger.info("Freezing embedding layer - embeddings will not be updated during training")
            model.embedding.weight.requires_grad = False
        
        # Increase dropout rates in CNN layers for better regularization
        model.spatial_dropout = nn.Dropout2d(0.4)  # Increase from 0.3
        model.dropout1 = nn.Dropout(0.6)  # Increase from 0.5
        
        return model.to(self.device)
    
    def train_fold(self, model, train_loader, val_loader, fold=None):
        """
        Train a model on the given fold data
        
        Args:
            model: The initialized model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            fold: Optional fold number for logging
            
        Returns:
            tuple: Training history and best model state dict
        """
        fold_str = f" (Fold {fold})" if fold is not None else ""
        logger.info(f"Starting training{fold_str} for {self.config['epochs']} epochs...")
        
        # Improved loss function with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer with increased weight decay for better regularization
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],  # Increased from 0.01
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing learning rate scheduler with warmup
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,  # First restart occurs after 5 epochs
            T_mult=1,
            eta_min=1e-6
        )
        
        # Track training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'best_val_loss': float('inf'),
            'best_val_acc': 0.0,
            'best_epoch': 0,
            'learning_rates': []
        }
        
        # Early stopping counter
        early_stop_counter = 0
        best_model_state = None
        
        # Training loop
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
                
                # Apply gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Track statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
                
                # Log batch progress (less frequently)
                if (batch_idx + 1) % 50 == 0:
                    logger.info(f'Epoch [{epoch+1}/{self.config["epochs"]}]{fold_str}, '
                                f'Batch [{batch_idx+1}/{len(train_loader)}], '
                                f'Loss: {loss.item():.4f}')
            
            # Update learning rate scheduler
            scheduler.step()
            
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
            
            # Track current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            
            # Log epoch stats
            logger.info(f'Epoch [{epoch+1}/{self.config["epochs"]}]{fold_str}, '
                       f'Train Loss: {epoch_train_loss:.4f}, '
                       f'Train Acc: {epoch_train_acc:.4f}, '
                       f'Val Loss: {epoch_val_loss:.4f}, '
                       f'Val Acc: {epoch_val_acc:.4f}, '
                       f'LR: {current_lr:.7f}')
            
            # Check for improvement
            improved = False
            
            # Check accuracy improvement (primary metric)
            if epoch_val_acc > history['best_val_acc']:
                logger.info(f'Validation accuracy improved from {history["best_val_acc"]:.4f} to {epoch_val_acc:.4f}')
                history['best_val_acc'] = epoch_val_acc
                improved = True
                
            # Check loss improvement (secondary metric)
            if epoch_val_loss < history['best_val_loss']:
                logger.info(f'Validation loss improved from {history["best_val_loss"]:.4f} to {epoch_val_loss:.4f}')
                history['best_val_loss'] = epoch_val_loss
                improved = True
            
            if improved:
                history['best_epoch'] = epoch + 1
                
                # Save best model state
                best_model_state = model.state_dict().copy()
                
                # Reset early stopping counter
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                logger.info(f'No improvement detected. Counter: {early_stop_counter}/{self.config["early_stopping_patience"]}')
                
                # Check early stopping
                if early_stop_counter >= self.config['early_stopping_patience']:
                    logger.info(f'Early stopping triggered after {epoch+1} epochs')
                    break
        
        # Training completed
        training_time = time.time() - start_time
        logger.info(f'Training{fold_str} completed in {training_time:.2f} seconds')
        logger.info(f'Best validation accuracy: {history["best_val_acc"]:.4f} at epoch {history["best_epoch"]}')
        
        return history, best_model_state
    
    def train_with_kfold(self):
        """
        Train the model using k-fold cross-validation
        
        Returns:
            tuple: List of fold histories and list of best models
        """
        try:
            # Load and preprocess data
            X, y, embedding_matrix, vocab_size, max_len = self.load_data()
            
            # Initialize k-fold cross-validation
            kf = KFold(n_splits=self.config['k_folds'], shuffle=True, random_state=self.config['random_state'])
            
            fold_histories = []
            fold_models = []
            fold_accuracies = []
            
            # Train on each fold
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                logger.info(f"Starting fold {fold+1}/{self.config['k_folds']}")
                
                # Split data for this fold
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                
                # Create data loaders
                train_loader, val_loader = self.create_data_loaders(X_train, y_train, X_val, y_val)
                logger.info(f"Fold {fold+1}: {len(train_idx)} training samples, {len(val_idx)} validation samples")
                
                # Initialize model for this fold
                model = self.init_model(vocab_size, embedding_matrix, max_len)
                
                # Train on this fold
                fold_history, fold_best_model = self.train_fold(model, train_loader, val_loader, fold+1)
                
                # Save the fold results
                fold_histories.append(fold_history)
                fold_models.append(fold_best_model)
                fold_accuracies.append(fold_history['best_val_acc'])
                
                # Save fold model
                fold_model_path = os.path.join(self.config['models_dir'], f'best_model_fold_{fold+1}.pth')
                torch.save(fold_best_model, fold_model_path)
                logger.info(f"Saved best model for fold {fold+1} to {fold_model_path}")
            
            # Compute and log average metrics across folds
            avg_accuracy = np.mean(fold_accuracies)
            std_accuracy = np.std(fold_accuracies)
            logger.info(f"Cross-validation complete. Average validation accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
            
            # Save the best fold's model as the final model
            best_fold_idx = np.argmax(fold_accuracies)
            final_model_path = os.path.join(self.config['models_dir'], 'best_model_kfold.pth')
            torch.save(fold_models[best_fold_idx], final_model_path)
            logger.info(f"Saved best model from fold {best_fold_idx+1} (acc: {fold_accuracies[best_fold_idx]:.4f}) to {final_model_path}")
            
            # Save fold metrics
            metrics_path = os.path.join(self.config['models_dir'], 'kfold_metrics.json')
            fold_metrics = {
                'fold_accuracies': [float(acc) for acc in fold_accuracies],
                'avg_accuracy': float(avg_accuracy),
                'std_accuracy': float(std_accuracy),
                'best_fold': int(best_fold_idx + 1)
            }
            with open(metrics_path, 'w') as f:
                json.dump(fold_metrics, f)
            logger.info(f"K-fold metrics saved to {metrics_path}")
            
            return fold_histories, fold_models
            
        except Exception as e:
            logger.error(f"Error during k-fold training: {e}")
            raise
    
    def train_single_split(self):
        """
        Train the model using a single train/validation split
        
        Returns:
            tuple: Trained model and training history
        """
        try:
            # Load and preprocess data
            X, y, embedding_matrix, vocab_size, max_len = self.load_data()
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, 
                test_size=self.config['validation_split'],
                random_state=self.config['random_state'],
                stratify=y
            )
            
            # Create data loaders
            train_loader, val_loader = self.create_data_loaders(X_train, y_train, X_val, y_val)
            logger.info(f"Data prepared: {len(X_train)} training samples, {len(X_val)} validation samples")
            
            # Initialize model
            model = self.init_model(vocab_size, embedding_matrix, max_len)
            
            # Train model
            history, best_model_state = self.train_fold(model, train_loader, val_loader)
            
            # Save best model
            best_model_path = os.path.join(self.config['models_dir'], 'best_model_single_split.pth')
            torch.save(best_model_state, best_model_path)
            logger.info(f'Best model saved to {best_model_path}')
            
            # Save training history
            history_path = os.path.join(self.config['models_dir'], 'training_history_single_split.json')
            with open(history_path, 'w') as f:
                json.dump({k: v for k, v in history.items() if isinstance(v, list) or isinstance(v, float) or isinstance(v, int)}, f)
            logger.info(f'Training history saved to {history_path}')
            
            return model, history
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def train(self):
        """
        Main training method that decides between k-fold and single split approaches
        
        Returns:
            tuple: Trained model(s) and training history/histories
        """
        if self.config['use_kfold']:
            logger.info(f"Starting {self.config['k_folds']}-fold cross-validation training")
            return self.train_with_kfold()
        else:
            logger.info("Starting single train/validation split training")
            return self.train_single_split()
    
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CNN BiLSTM classifier with improvements")
    parser.add_argument('--config', type=str, default=None, 
                        help='Path to JSON configuration file')
    parser.add_argument('--kfold', action='store_true',
                        help='Use k-fold cross-validation')
    parser.add_argument('--folds', type=int, default=3,
                        help='Number of folds for cross-validation')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--embedding-dim', type=int, default=100,
                        help='Dimension of word embeddings')
                        
    args = parser.parse_args()
    
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
            logger.info(f"Loaded configuration from {args.config}")
    else:
        # Create config from command line arguments
        config = {
            'use_kfold': args.kfold,
            'k_folds': args.folds,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'embedding_dim': args.embedding_dim
        }
    
    logger.info("Starting improved CNN BiLSTM training")
    
    # Create trainer instance
    trainer = ImprovedCNNBiLSTMTrainer(config)
    
    try:
        # Train model
        results = trainer.train()
        
        if trainer.config['use_kfold']:
            fold_histories, fold_models = results
            # Results already logged in train_with_kfold method
        else:
            model, history = results
            logger.info(f"Best validation accuracy: {history.get('best_val_acc', 0):.4f} at epoch {history['best_epoch']}")
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
