from src.common_imports import * # noqa: F403, F405
from src.logging_config import *  # noqa: F403, F405
from src.model_builder import ModelBuilder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import json
import logging
from tqdm import tqdm
import gc

class ModelTrainer:
    """Class for training PyTorch models for sentiment analysis"""
    
    def __init__(self, save_data_dir):
        """
        Initialize the ModelTrainer
        
        Args:
            save_data_dir (str): Directory to save model files and training history
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.SAVE_DATA_DIR = save_data_dir
        self.model_builder = ModelBuilder(save_data_dir)
    
    def train_model(self, X, y, embedding_matrix, max_len, model_type='ensemble'):
        """
        Train the PyTorch model with improved progress tracking
        
        Args:
            X (numpy.ndarray): Input features
            y (numpy.ndarray): Target labels
            embedding_matrix (numpy.ndarray): Word embedding matrix
            max_len (int): Maximum sequence length
            model_type (str): Type of model ('simple', 'deep', 'stacked', or 'ensemble')
            
        Returns:
            tuple: (model, history) - Trained model and training history
        """
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
        
        # Build model
        model = self.model_builder.get_model(
            input_size=len(embedding_matrix), 
            embedding_dim=embedding_matrix.shape[1],
            model_type=model_type
        )
        
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
    
    def save_training_history(self, history, model_type):
        """
        Save training history to file
        
        Args:
            history (dict): Dictionary containing training metrics
            model_type (str): Type of model
        """
        history_path = os.path.join(self.SAVE_DATA_DIR, f'training_history_{model_type}.json')
        with open(history_path, 'w') as f:
            # Convert tensors/numpy arrays to Python lists for JSON serialization
            serializable_history = {}
            for key, values in history.items():
                serializable_history[key] = [float(val) for val in values]
            json.dump(serializable_history, f)
    
    def _plot_training_history(self, history, model_type):
        """
        Create and save plots of training history
        
        Args:
            history (dict): Dictionary containing training metrics
            model_type (str): Type of model
        """
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
