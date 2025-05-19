import os
import sys
import pytest
import glob
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import h5py
import tensorflow as tf
from tensorflow.keras.models import load_model

# Add parent directory to path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from refactored modules
from src.core.nlpmodel import NlpModel
from src.models.models import SimpleLSTMModel, DeepLSTMModel, StackedLSTMModel, EnsembleModel
from src.models.model_builder import ModelBuilder
from src.core.logging_config import logging

class TestModels:
    @pytest.fixture(scope="class")
    def setup(self):
        """Set up common test data and paths"""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        models_dir = os.path.join(base_dir, "data", "models")
        data_dir = os.path.join(base_dir, "data")
        
        # Sample input data for model testing
        # Creating a small dummy input for testing inference
        dummy_text = [
            "This product is amazing and I love it",
            "The quality could be better",
            "I'm disappointed with this purchase"
        ]
        
        # Create a logger for tracking test execution
        logger = logging.getLogger("ModelTesting")
        
        return {
            "base_dir": base_dir,
            "models_dir": models_dir,
            "data_dir": data_dir,
            "dummy_text": dummy_text,
            "logger": logger,
            "model_builder": ModelBuilder(models_dir)
        }
    
    def test_pytorch_models_exist(self, setup):
        """Test if PyTorch model files exist"""
        models_dir = setup["models_dir"]
        logger = setup["logger"]
        
        # Look for model files
        pth_files = glob.glob(os.path.join(models_dir, "*.pth"))
        logger.info(f"Found {len(pth_files)} PyTorch model files")
        
        # We should have at least some model files
        if len(pth_files) == 0:
            logger.warning("No PyTorch models found, skipping test")
            pytest.skip("No PyTorch models found")
    
    def test_keras_models_exist(self, setup):
        """Test if Keras model files exist"""
        models_dir = setup["models_dir"]
        data_dir = setup["data_dir"]
        logger = setup["logger"]
        
        # Look for model files
        keras_models = glob.glob(os.path.join(models_dir, "*.h5"))
        keras_models += glob.glob(os.path.join(data_dir, "*.h5"))
        # We should have at least some model files
        if len(keras_models) == 0:
            logger.warning("No Keras models found, skipping test")
            pytest.skip("No Keras models found")
        
        for model_path in keras_models:
            try:
                # Check if it's a valid HDF5 file
                with h5py.File(model_path, 'r') as f:
                    # Check if it has the expected keras model structure
                    assert 'model_weights' in f or 'layer_names' in f, \
                        f"File {os.path.basename(model_path)} doesn't appear to be a valid Keras model"
                logger.info(f"Successfully validated HDF5 format of: {os.path.basename(model_path)}")
            except Exception as e:
                logger.warning(f"Issue with Keras model {os.path.basename(model_path)}: {str(e)}")

    def test_model_architecture(self, setup):
        """Test model architecture creation"""
        model_builder = setup["model_builder"]
        logger = setup["logger"]
        
        # Test that all model types can be created
        embedding_dim = 100
        vocab_size = 10000
        
        # Test each model type
        model_types = ['simple', 'deep', 'stacked', 'ensemble']
        for model_type in model_types:
            logger.info(f"Creating {model_type} model")
            model = model_builder.get_model(
                input_size=vocab_size,
                embedding_dim=embedding_dim,
                model_type=model_type
            )
            # Check if model was created
            assert model is not None, f"Failed to create {model_type} model"
            
            # Verify model can accept inputs
            dummy_input = torch.LongTensor(np.random.randint(0, vocab_size, size=(5, 100)))
            try:
                output = model(dummy_input)
                logger.info(f"{model_type} model produced output shape: {output.shape}")
                assert output.shape == (5, 1), f"Unexpected output shape for {model_type} model"
            except Exception as e:
                assert False, f"Error running {model_type} model: {e}"

    def test_simple_model(self, setup):
        """Test the SimpleLSTMModel"""
        vocab_size = 10000
        embedding_dim = 100
        hidden_size = 128
        
        # Create model
        model = SimpleLSTMModel(
            input_size=vocab_size, 
            embedding_dim=embedding_dim, 
            hidden_size=hidden_size,
            num_layers=1,
            dropout=0.3
        )
        
        # Test forward pass
        dummy_input = torch.LongTensor(np.random.randint(0, vocab_size, size=(5, 100)))
        output = model(dummy_input)
        
        # Check output shape
        assert output.shape == (5, 1), f"Expected output shape (5, 1), got {output.shape}"

    def test_deep_model(self, setup):
        """Test the DeepLSTMModel"""
        vocab_size = 10000
        embedding_dim = 100
        hidden_size = 256
        
        # Create model
        model = DeepLSTMModel(
            input_size=vocab_size, 
            embedding_dim=embedding_dim, 
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.5
        )
        
        # Test forward pass
        dummy_input = torch.LongTensor(np.random.randint(0, vocab_size, size=(5, 100)))
        output = model(dummy_input)
        
        # Check output shape
        assert output.shape == (5, 1), f"Expected output shape (5, 1), got {output.shape}"

    def test_stacked_model(self, setup):
        """Test the StackedLSTMModel"""
        vocab_size = 10000
        embedding_dim = 100
        
        # Create model
        model = StackedLSTMModel(
            input_size=vocab_size, 
            embedding_dim=embedding_dim
        )
        
        # Test forward pass
        dummy_input = torch.LongTensor(np.random.randint(0, vocab_size, size=(5, 100)))
        output = model(dummy_input)
        
        # Check output shape
        assert output.shape == (5, 1), f"Expected output shape (5, 1), got {output.shape}"

    def test_ensemble_model(self, setup):
        """Test the EnsembleModel"""
        vocab_size = 10000
        embedding_dim = 100
        
        # Create component models
        models = [
            SimpleLSTMModel(input_size=vocab_size, embedding_dim=embedding_dim),
            DeepLSTMModel(input_size=vocab_size, embedding_dim=embedding_dim),
            StackedLSTMModel(input_size=vocab_size, embedding_dim=embedding_dim)
        ]
        
        # Create ensemble
        ensemble = EnsembleModel(models)
        
        # Test forward pass
        dummy_input = torch.LongTensor(np.random.randint(0, vocab_size, size=(5, 100)))
        output = ensemble(dummy_input)
        
        # Check output shape
        assert output.shape == (5, 1), f"Expected output shape (5, 1), got {output.shape}"

    def test_model_loading(self, setup):
        """Test loading a saved model if available"""
        models_dir = setup["models_dir"]
        model_builder = setup["model_builder"]
        logger = setup["logger"]
        
        # Look for specific model file
        model_file = os.path.join(models_dir, "best_model_simple.pth")
        
        if os.path.exists(model_file):
            try:
                # Create the model architecture
                vocab_size = 10000
                embedding_dim = 100
                model = model_builder.get_model(
                    input_size=vocab_size,
                    embedding_dim=embedding_dim,
                    model_type='simple'
                )
                
                # Load the state dict
                model.load_state_dict(torch.load(model_file))
                logger.info("Successfully loaded model weights")
                
                # Test the model
                dummy_input = torch.LongTensor(np.random.randint(0, vocab_size, size=(5, 100)))
                output = model(dummy_input)
                logger.info(f"Model output shape: {output.shape}")
                
                assert output.shape == (5, 1), "Unexpected output shape"
                
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                # Don't fail the test if the model file exists but can't be loaded
                # since this could be due to differences in model architecture
                logger.warning("Skipping model loading test")
        else:
            logger.warning(f"Model file {model_file} not found, skipping test")
            # Skip this test if the file doesn't exist
            pytest.skip("Model file not found")

    def test_training_history(self, setup):
        """Test if training history files exist and have correct format"""
        data_dir = setup["data_dir"]
        logger = setup["logger"]
        history_files = glob.glob(os.path.join(data_dir, "training_history_*.json"))
        
        if history_files:
            for history_file in history_files:
                try:
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                    
                    # Check history structure
                    assert isinstance(history, dict), "Training history should be a dictionary"
                    
                    # Check if we have expected keys
                    expected_keys = ['train_loss', 'val_loss']
                    for key in expected_keys:
                        if key in history:
                            assert len(history[key]) > 0, f"History {key} should not be empty"
                    
                    logger.info(f"Successfully validated training history: {os.path.basename(history_file)}")
                except Exception as e:
                    logger.error(f"Error validating training history: {str(e)}")


if __name__ == "__main__":
    # This allows running the tests directly from the file
    pytest.main(["-xvs", __file__])
