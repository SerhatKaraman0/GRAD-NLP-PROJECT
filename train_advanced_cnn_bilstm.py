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
            'dropout_rate': 0.5,
            'spatial_dropout': 0.4,
            'lstm_hidden_size': 256,
            'lstm_layers': 2,
            'weight_decay': 1e-4,
            'gradient_clip': 1.0,
            'label_smoothing': 0.2,
            'use_mixed_precision': True,
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
            
            # One-cycle learning rate scheduler for improved convergence
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config['learning_rate'],
                steps_per_epoch=len(train_loader),
                epochs=self.config['epochs'],
                pct_start=0.3,
                div_factor=25.0,
                final_div_factor=10000.0,
                anneal_strategy='cos'
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
                    
                    # Update learning rate with one-cycle policy
                    scheduler.step()
                    
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
    
    # Print configuration summary
    logger.info("Training configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Create trainer with config
    trainer = AdvancedCNNBiLSTMTrainer(config)
    
    try:
        # Train model
        model, history = trainer.train()
        
        logger.info(f"Best validation accuracy: {history['best_val_acc']:.4f} at epoch {history['best_epoch']}")
        logger.info(f"Best validation loss: {history['best_val_loss']:.4f} at epoch {history['best_epoch']}")
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
