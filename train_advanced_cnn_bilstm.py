#!/usr/bin/env python
# filepath: /Users/user/Desktop/Projects/NLP-Learning/train_advanced_cnn_bilstm.py

"""
Enhanced training script for the Advanced CNN-BiLSTM model.
This script uses state-of-the-art training techniques to improve model performance.
"""

import os
import sys
import argparse
import json
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import project modules
from src.features.download_embeddings import ensure_glove_embeddings
from src.models.train_cnn_lstm import CNNBiLSTMTrainer
from src.models.advanced_models import AdvancedCNNBiLSTMClassifier
from src.core.logging_config import setup_logging, get_logger

# Setup logger
setup_logging()
logger = get_logger('Advanced_CNNBiLSTM_Training')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train the Advanced CNN BiLSTM classifier model')
    
    parser.add_argument('--config', type=str, default='config/advanced_cnn_bilstm.json',
                        help='Path to JSON configuration file')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--kfold', action='store_true',
                        help='Enable k-fold cross-validation (overrides config setting)')
    parser.add_argument('--no-kfold', action='store_true',
                        help='Disable k-fold cross-validation (overrides config setting)')
    parser.add_argument('--folds', type=int, default=None,
                        help='Number of folds for cross-validation (overrides config setting)')
    
    return parser.parse_args()

class AdvancedCNNBiLSTMTrainer(CNNBiLSTMTrainer):
    """
    Extended trainer with advanced training techniques
    """
    def __init__(self, config=None):
        """Initialize with enhanced configuration"""
        super().__init__(config)
        
        # Override configuration with advanced parameters
        advanced_defaults = {
            'dropout_rate': 0.6,
            'spatial_dropout': 0.5,
            'lstm_hidden_size': 256,
            'lstm_layers': 2,
            'weight_decay': 0.0003,
            'gradient_clip': 1.0,
            'label_smoothing': 0.2,
            'use_mixed_precision': True,
            'use_kfold': True,
            'k_folds': 5
        }
        
        # Set advanced defaults if not provided in config
        for key, value in advanced_defaults.items():
            if key not in self.config:
                self.config[key] = value
                
        # Configure mixed precision training if available
        self.scaler = None
        if self.config['use_mixed_precision'] and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Using mixed precision training")
    
    def load_data(self, return_raw_data=False):
        """
        Load data for training/validation
        
        Args:
            return_raw_data: If True, return raw data for k-fold cross-validation
            
        Returns:
            If return_raw_data is False: 
                tuple: (train_loader, val_loader, vocab_size, embedding_matrix, max_len)
            Else:
                tuple: (X, y, embedding_matrix, vocab_size, max_len)
        """
        try:
            from src.features.embeddings import EmbeddingProcessor
            
            logger.info(f"Loading data from {self.config['data_path']}")
            import pandas as pd
            df = pd.read_csv(self.config['data_path'])
            
            # Ensure text values are strings
            if 'text_column' in self.config:
                text_column = self.config['text_column']
            else:
                text_column = 'Text'  # Default column name
            
            df[text_column] = df[text_column].astype(str)
            
            # Initialize embedding processor
            embedding_processor = EmbeddingProcessor(
                embedding_dim=self.config['embedding_dim'], 
                max_features=self.config['max_features']
            )
            
            # Convert text to sequences and get pre-trained embedding matrix
            X_sequences, embedding_matrix = embedding_processor.prepare_embeddings(df[text_column].tolist())
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
            
            if return_raw_data:
                return X_padded, y, embedding_matrix, embedding_processor.max_features, max_len
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_padded, y, 
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
            return train_loader, val_loader, embedding_processor.max_features, embedding_matrix, max_len
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def train(self):
        """
        Train the Advanced CNN-BiLSTM classifier model with enhanced techniques
        
        Returns:
            tuple: Trained model and training history
        """
        try:
            # Load data
            train_loader, val_loader, vocab_size, embedding_matrix, max_len = self.load_data()
            
            # Initialize advanced model
            model = AdvancedCNNBiLSTMClassifier(
                input_size=max_len,
                embedding_dim=self.config['embedding_dim'],
                vocab_size=vocab_size + 1,  # +1 for padding token
                num_classes=self.config['num_classes'],
                dropout_rate=self.config['dropout_rate'],
                spatial_dropout=self.config['spatial_dropout'],
                lstm_hidden_size=self.config['lstm_hidden_size'],
                lstm_layers=self.config['lstm_layers']
            )
            
            logger.info(f"Initialized Advanced CNN-BiLSTM with input size {max_len}")
            
            # Initialize embedding layer with pre-trained weights
            logger.info(f"Initializing embedding layer with pre-trained GloVe {self.config['embedding_dim']}d embeddings")
            
            # Add a row of zeros at the beginning for the padding token (index 0)
            padded_embedding_matrix = torch.zeros((vocab_size + 1, self.config['embedding_dim']))
            padded_embedding_matrix[1:] = torch.from_numpy(embedding_matrix[:vocab_size])
            
            # Copy the padded embedding matrix to the model's embedding layer
            model.embedding.weight.data.copy_(padded_embedding_matrix)
            
            # Option to freeze the embedding layer
            freeze_embeddings = self.config.get('freeze_embeddings', False)
            if freeze_embeddings:
                logger.info("Freezing embedding layer - embeddings will not be updated during training")
                model.embedding.weight.requires_grad = False
            
            model = model.to(self.device)
            
            # Advanced loss function with label smoothing
            criterion = torch.nn.CrossEntropyLoss(
                label_smoothing=self.config['label_smoothing']
            )
            
            # Enhanced optimizer with weight decay
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay'],
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            # Cosine annealing learning rate scheduler with warmup
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
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
            
            # Early stopping counter with patience
            early_stop_counter = 0
            
            # Training loop with improved techniques
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
                    
                    # Forward pass with mixed precision if enabled
                    if self.scaler:
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                        
                        # Backward pass with gradient scaling
                        self.scaler.scale(loss).backward()
                        
                        # Gradient clipping to prevent exploding gradients
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['gradient_clip'])
                        
                        # Update weights
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        # Standard forward pass
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        
                        # Backward pass
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['gradient_clip'])
                        
                        # Update weights
                        optimizer.step()
                    
                    # Note: For CosineAnnealingWarmRestarts, we step the scheduler once per epoch, not per batch
                    
                    # Track statistics
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += targets.size(0)
                    train_correct += (predicted == targets).sum().item()
                    
                    # Log batch progress (less frequently to reduce clutter)
                    if (batch_idx + 1) % 50 == 0:
                        logger.info(f'Epoch [{epoch+1}/{self.config["epochs"]}], '
                                    f'Batch [{batch_idx+1}/{len(train_loader)}], '
                                    f'Loss: {loss.item():.4f}')
                
                # Compute epoch training stats
                epoch_train_loss = train_loss / len(train_loader)
                epoch_train_acc = train_correct / train_total
                
                # Update learning rate scheduler after each epoch
                scheduler.step()
                
                # Validation phase
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        
                        # Use mixed precision for validation as well if enabled
                        if self.scaler:
                            with torch.cuda.amp.autocast():
                                outputs = model(inputs)
                                loss = criterion(outputs, targets)
                        else:
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
                
                # Track current learning rate (use last LR from scheduler)
                current_lr = optimizer.param_groups[0]['lr']
                history['learning_rates'].append(current_lr)
                
                # Log epoch stats
                logger.info(f'Epoch [{epoch+1}/{self.config["epochs"]}], '
                           f'Train Loss: {epoch_train_loss:.4f}, '
                           f'Train Acc: {epoch_train_acc:.4f}, '
                           f'Val Loss: {epoch_val_loss:.4f}, '
                           f'Val Acc: {epoch_val_acc:.4f}, '
                           f'LR: {current_lr:.7f}')
                
                # Check for improvement in both loss and accuracy
                improved_loss = epoch_val_loss < history['best_val_loss']
                improved_acc = epoch_val_acc > history['best_val_acc']
                
                if improved_loss or improved_acc:
                    improvement_message = []
                    if improved_loss:
                        improvement_message.append(f'val_loss: {history["best_val_loss"]:.4f} → {epoch_val_loss:.4f}')
                        history['best_val_loss'] = epoch_val_loss
                    
                    if improved_acc:
                        improvement_message.append(f'val_acc: {history["best_val_acc"]:.4f} → {epoch_val_acc:.4f}')
                        history['best_val_acc'] = epoch_val_acc
                    
                    history['best_epoch'] = epoch + 1
                    logger.info(f'Improvement detected: {", ".join(improvement_message)}')
                    
                    # Save best model
                    best_model_path = os.path.join(self.config['models_dir'], 'best_model_advanced_cnn_bilstm.pth')
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f'Best model saved to {best_model_path}')
                    
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
            logger.info(f'Training completed in {training_time:.2f} seconds')
            
            # Save final model
            final_model_path = os.path.join(self.config['models_dir'], 'final_model_advanced_cnn_bilstm.pth')
            torch.save(model.state_dict(), final_model_path)
            logger.info(f'Final model saved to {final_model_path}')
            
            # Save training history
            history_path = os.path.join(self.config['models_dir'], 'training_history_advanced_cnn_bilstm.json')
            with open(history_path, 'w') as f:
                json.dump({k: v for k, v in history.items() if isinstance(v, list)}, f)
            logger.info(f'Training history saved to {history_path}')
            
            return model, history
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def train_with_kfold(self):
        """
        Train the model using k-fold cross-validation
        
        Returns:
            tuple: List of fold histories and list of best models
        """
        try:
            # Load and preprocess data
            X, y, embedding_matrix, vocab_size, max_len = self.load_data(return_raw_data=True)
            
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
                
                logger.info(f"Fold {fold+1}: {len(train_idx)} training samples, {len(val_idx)} validation samples")
                
                # Initialize advanced model
                model = AdvancedCNNBiLSTMClassifier(
                    input_size=max_len,
                    embedding_dim=self.config['embedding_dim'],
                    vocab_size=vocab_size + 1,  # +1 for padding token
                    num_classes=self.config['num_classes'],
                    dropout_rate=self.config['dropout_rate'],
                    spatial_dropout=self.config['spatial_dropout'],
                    lstm_hidden_size=self.config['lstm_hidden_size'],
                    lstm_layers=self.config['lstm_layers']
                )
                
                logger.info(f"Initialized Advanced CNN-BiLSTM with input size {max_len} for fold {fold+1}")
                
                # Initialize embedding layer with pre-trained weights
                logger.info(f"Initializing embedding layer with pre-trained GloVe {self.config['embedding_dim']}d embeddings")
                
                # Add a row of zeros at the beginning for the padding token (index 0)
                padded_embedding_matrix = torch.zeros((vocab_size + 1, self.config['embedding_dim']))
                padded_embedding_matrix[1:] = torch.from_numpy(embedding_matrix[:vocab_size])
                
                # Copy the padded embedding matrix to the model's embedding layer
                model.embedding.weight.data.copy_(padded_embedding_matrix)
                
                # Option to freeze the embedding layer
                freeze_embeddings = self.config.get('freeze_embeddings', False)
                if freeze_embeddings:
                    logger.info("Freezing embedding layer - embeddings will not be updated during training")
                    model.embedding.weight.requires_grad = False
                
                model = model.to(self.device)
                
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
        
        # Advanced loss function with label smoothing
        criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=self.config['label_smoothing']
        )
        
        # Enhanced optimizer with weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine annealing learning rate scheduler with warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
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
                
                # Forward pass with mixed precision if enabled
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['gradient_clip'])
                    
                    # Update weights
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # Standard forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['gradient_clip'])
                    
                    # Update weights
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
            
            # Update learning rate scheduler after each epoch
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
                    
                    # Use mixed precision for validation as well if enabled
                    if self.scaler:
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                    else:
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

def main():
    """Main function to run training"""
    args = parse_arguments()
    logger.info("Starting Advanced CNN BiLSTM model training")
    
    # Create directories if they don't exist
    os.makedirs('config', exist_ok=True)
    
    # Ensure GloVe embeddings are available
    logger.info("Checking for GloVe embeddings...")
    if not ensure_glove_embeddings():
        logger.error("Failed to setup GloVe embeddings. Exiting.")
        sys.exit(1)
    logger.info("GloVe embeddings are available")
    
    # Load configuration
    if not os.path.exists(args.config):
        logger.error(f"Configuration file {args.config} not found. Please run with a valid config path.")
        sys.exit(1)
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Override CUDA setting if needed
    if args.no_cuda:
        config['use_cuda'] = False
    elif 'use_cuda' not in config:
        config['use_cuda'] = torch.cuda.is_available()
    
    # Override k-fold settings if specified in command line
    if args.kfold:
        config['use_kfold'] = True
    elif args.no_kfold:
        config['use_kfold'] = False
    
    if args.folds is not None:
        config['k_folds'] = args.folds
    
    # Print configuration summary
    logger.info("Training configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Create trainer with config
    trainer = AdvancedCNNBiLSTMTrainer(config)
    
    try:
        # Train model based on selected approach
        if config.get('use_kfold', False):
            logger.info(f"Using {config['k_folds']}-fold cross-validation for training")
            fold_histories, fold_models = trainer.train_with_kfold()
            logger.info("K-fold cross-validation training completed successfully")
        else:
            logger.info("Using single train/validation split for training")
            model, history = trainer.train()
            logger.info(f"Best validation accuracy: {history['best_val_acc']:.4f} at epoch {history['best_epoch']}")
            logger.info(f"Best validation loss: {history['best_val_loss']:.4f} at epoch {history['best_epoch']}")
            logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
