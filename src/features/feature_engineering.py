"""
Legacy module that now serves as a wrapper around the refactored modules.
This file is kept for backward compatibility, but all functionality
has been moved to specialized modules:
- data_processing.py
- embeddings.py 
- model_builder.py
- model_training.py
- model_evaluation.py
- dashboard_generator.py
"""

from src.core.common_imports import * # noqa: F403, F405
from src.core.logging_config import *  # noqa: F403, F405
from src.core.nlpmodel import NlpModel
from src.data.data_processing import DataProcessor
from src.features.embeddings import EmbeddingProcessor
from src.models.model_builder import ModelBuilder
from src.models.model_training import ModelTrainer
from src.models.model_evaluation import ModelEvaluator
from src.visualization.dashboard_generator import DashboardGenerator

import os
import gc
import sys
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class FeatureEngineering(NlpModel):
    """
    Legacy class that delegates to specialized modules.
    Maintained for backward compatibility with existing code.
    """
    def __init__(self, batch_size=10000, max_features=10000, embedding_dim=100):
        """
        Initialize the FeatureEngineering class with components from the refactored modules
        
        Args:
            batch_size (int): Size of batch for processing
            max_features (int): Maximum features for vectorization
            embedding_dim (int): Embedding dimension
        """
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.SAVE_DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.STATS_DIR = os.path.join(self.BASE_DIR, "stats")
        self.batch_size = batch_size
        self.max_features = max_features
        self.embedding_dim = embedding_dim
        
        # Initialize specialized components
        self.data_processor = DataProcessor(batch_size=batch_size, max_features=max_features)
        self.embedding_processor = EmbeddingProcessor(batch_size=batch_size, max_features=max_features, embedding_dim=embedding_dim)
        self.model_builder = ModelBuilder(self.SAVE_DATA_DIR)
        self.model_trainer = ModelTrainer(self.SAVE_DATA_DIR)
        self.model_evaluator = ModelEvaluator(self.SAVE_DATA_DIR)
        self.dashboard_generator = DashboardGenerator(self.SAVE_DATA_DIR, max_features=max_features, embedding_dim=embedding_dim)
        
        self.logger.info("FeatureEngineering initialized with refactored components")

    # Forward methods to specialized components
    def word_freq(self):
        """Forward to data processor"""
        return self.data_processor.word_freq()

    def create_bow(self):
        """Forward to data processor"""
        return self.data_processor.create_bow()

    def load_and_process_data(self):
        """Forward to embedding processor"""
        return self.embedding_processor.load_and_process_data()
    
    def prepare_embeddings(self, texts):
        """Forward to embedding processor"""
        return self.embedding_processor.prepare_embeddings(texts)
    
    def build_lstm_model(self, input_size, embedding_dim, hidden_size, num_layers, dropout):
        """Forward to model builder"""
        return self.model_builder.build_lstm_model(input_size, embedding_dim, hidden_size, num_layers, dropout)
    
    def build_stacked_lstm_model(self, input_size, embedding_dim):
        """Forward to model builder"""
        return self.model_builder.build_stacked_lstm_model(input_size, embedding_dim)
        
    def train_model(self, X, y, embedding_matrix, max_len, model_type='ensemble'):
        """Forward to model trainer"""
        return self.model_trainer.train_model(X, y, embedding_matrix, max_len, model_type)
        
    def evaluate_model(self, model, X_test, y_test, model_type='ensemble'):
        """Forward to model evaluator"""
        return self.model_evaluator.evaluate_model(model, X_test, y_test, model_type)
    
    def generate_model_dashboard(self, model_results, model_type='ensemble'):
        """Forward to dashboard generator"""
        return self.dashboard_generator.generate_model_dashboard(model_results, model_type)
    
    def save_to_parquet(self, df, output_path: str = "processed_data.parquet") -> None:
        """Save the processed DataFrame to Parquet with gzip compression"""
        self.logger.info("SAVING TO PARQUET STARTED..")
        
        # Check if DataFrame has sparse data
        has_sparse = hasattr(df, 'sparse') and hasattr(df.sparse, 'to_dense')
    
        if has_sparse:
            # Convert sparse DataFrame to dense
            dense_df = df.sparse.to_dense()
        else:
            self.logger.warning("Input DataFrame does not contain sparse data.")
            dense_df = df
    
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
        # Save to Parquet format with gzip compression
        dense_df.to_parquet(f"{output_path}.parquet.gz", compression="gzip")
        self.logger.info(f"DF SAVED TO {output_path}.parquet.gz")
        return output_path


def main():
    """Main execution function."""
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
            model.logger.info(f"\n{'='*50}")
            model.logger.info(f"Training {model_type.upper()} LSTM Model")            
            model.logger.info(f"{'='*50}")
            
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
            
        # Generate combined dashboard
        model.dashboard_generator.generate_combined_dashboard(results)
        
    except Exception as e:
        model.logger.error(f"Error in main execution: {e}")
        import traceback
        model.logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
