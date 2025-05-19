from src.core.common_imports import * # noqa: F403, F405
from src.core.logging_config import *  # noqa: F403, F405
from src.models.models import SimpleLSTMModel, DeepLSTMModel, StackedLSTMModel, EnsembleModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import logging
import os

class ModelBuilder:
    """Class for building various types of LSTM models for sentiment analysis"""
    
    def __init__(self, save_data_dir):
        """
        Initialize the ModelBuilder
        
        Args:
            save_data_dir (str): Directory to save model files
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.SAVE_DATA_DIR = save_data_dir
        
    def build_lstm_model(self, input_size, embedding_dim, hidden_size, num_layers, dropout):
        """
        Build a PyTorch LSTM model
        
        Args:
            input_size (int): Size of the input vocabulary
            embedding_dim (int): Dimension of the embeddings
            hidden_size (int): Size of the hidden layer
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
            
        Returns:
            torch.nn.Module: The LSTM model
        """
        return SimpleLSTMModel(
            input_size=input_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def build_deep_lstm_model(self, input_size, embedding_dim):
        """
        Build a deep LSTM model with more layers
        
        Args:
            input_size (int): Size of the input vocabulary
            embedding_dim (int): Dimension of the embeddings
            
        Returns:
            torch.nn.Module: The deep LSTM model
        """
        return DeepLSTMModel(
            input_size=input_size, 
            embedding_dim=embedding_dim, 
            hidden_size=256, 
            num_layers=2, 
            dropout=0.5
        )
    
    def build_stacked_lstm_model(self, input_size, embedding_dim):
        """
        Build a stacked LSTM model with multiple LSTM layers
        
        Args:
            input_size (int): Size of the input vocabulary
            embedding_dim (int): Dimension of the embeddings
            
        Returns:
            torch.nn.Module: The stacked LSTM model
        """
        return StackedLSTMModel(input_size=input_size, embedding_dim=embedding_dim)
    
    def build_ensemble_model(self, input_size, embedding_dim):
        """
        Build an ensemble of different LSTM models
        
        Args:
            input_size (int): Size of the input vocabulary
            embedding_dim (int): Dimension of the embeddings
            
        Returns:
            list: List of models that form the ensemble
        """
        # Create ensemble of models
        models = []
        
        # Simple LSTM
        models.append(self.build_lstm_model(
            input_size=input_size, 
            embedding_dim=embedding_dim, 
            hidden_size=128, 
            num_layers=1, 
            dropout=0.3
        ))
        
        # Deep LSTM
        models.append(self.build_lstm_model(
            input_size=input_size, 
            embedding_dim=embedding_dim, 
            hidden_size=256, 
            num_layers=2, 
            dropout=0.5
        ))
        
        # Stacked LSTM
        models.append(self.build_stacked_lstm_model(
            input_size=input_size, 
            embedding_dim=embedding_dim
        ))
        
        return models
    
    def get_model(self, input_size, embedding_dim, model_type='ensemble'):
        """
        Factory method to get a model of the specified type
        
        Args:
            input_size (int): Size of the input vocabulary
            embedding_dim (int): Dimension of the embeddings
            model_type (str): Type of model ('simple', 'deep', 'stacked', or 'ensemble')
            
        Returns:
            torch.nn.Module: The requested model
        """
        if model_type == 'simple':
            model = self.build_lstm_model(
                input_size=input_size, 
                embedding_dim=embedding_dim, 
                hidden_size=128, 
                num_layers=1, 
                dropout=0.3
            )
        elif model_type == 'deep':
            model = self.build_lstm_model(
                input_size=input_size, 
                embedding_dim=embedding_dim, 
                hidden_size=256, 
                num_layers=2, 
                dropout=0.5
            )
        elif model_type == 'stacked':
            model = self.build_stacked_lstm_model(
                input_size=input_size, 
                embedding_dim=embedding_dim
            )
        elif model_type == 'ensemble':
            models = self.build_ensemble_model(input_size, embedding_dim)
            model = EnsembleModel(models)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
        
    def load_model(self, input_size, embedding_dim, model_type='ensemble', load_best=True):
        """
        Load a saved model from disk
        
        Args:
            input_size (int): Size of the input vocabulary
            embedding_dim (int): Dimension of the embeddings
            model_type (str): Type of model ('simple', 'deep', 'stacked', or 'ensemble')
            load_best (bool): Whether to load the best model or the latest
            
        Returns:
            torch.nn.Module: The loaded model
        """
        # Create a new model of the specified type
        model = self.get_model(input_size, embedding_dim, model_type)
        
        # Determine the path to the saved model
        prefix = 'best_model' if load_best else 'latest_model'
        model_path = os.path.join(self.SAVE_DATA_DIR, f'{prefix}_{model_type}.pth')
        
        # Check if the model file exists
        if not os.path.exists(model_path):
            self.logger.warning(f"Model file not found at {model_path}")
            return model
        
        # Load the saved state
        try:
            model.load_state_dict(torch.load(model_path))
            self.logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
        
        return model
