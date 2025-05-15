from src.common_imports import * # noqa: F403, F405
from src.logging_config import *  # noqa: F403, F405
from src.data_processing import DataProcessor
from src.embeddings import EmbeddingProcessor
from src.model_builder import ModelBuilder
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.dashboard_generator import DashboardGenerator

import os
import gc
import torch
import numpy as np
import logging
from sklearn.metrics import precision_score, recall_score, f1_score

# Set up logger
logger = logging.getLogger("Main")

def main():
    """Main execution function."""
    # Check if PyTorch GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Configuration parameters
    batch_size = 10000
    max_features = 7000
    embedding_dim = 100
    
    # Base directories
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    save_data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(save_data_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    try:
        # Initialize processors
        embedding_processor = EmbeddingProcessor(
            batch_size=batch_size, 
            max_features=max_features, 
            embedding_dim=embedding_dim
        )
        
        # Load and process data
        texts, labels = embedding_processor.load_and_process_data()
        
        # Prepare data for training
        X_train, y_train, X_test, y_test, embedding_matrix, max_len = embedding_processor.prepare_data_for_training(texts, labels)
        
        # Initialize trainers and evaluators
        model_trainer = ModelTrainer(models_dir)
        model_evaluator = ModelEvaluator(save_data_dir)
        dashboard_generator = DashboardGenerator(
            save_data_dir, 
            max_features=max_features, 
            embedding_dim=embedding_dim
        )
        
        # Train and evaluate each model type separately
        model_types = ['simple', 'deep', 'stacked', 'ensemble']
        results = {}
        
        for model_type in model_types:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_type.upper()} LSTM Model")            
            logger.info(f"{'='*50}")
            
            # Train model with PyTorch
            trained_model, history = model_trainer.train_model(
                X_train, y_train, 
                embedding_matrix, max_len,
                model_type=model_type
            )
            
            # Evaluate model
            mse, mae, acc, y_pred, detailed_results = model_evaluator.evaluate_model(
                trained_model, X_test, y_test, 
                model_type=model_type
            )
            
            # Store results
            results[model_type] = detailed_results
            
            # Generate dashboard
            dashboard_generator.generate_model_dashboard(detailed_results, model_type=model_type)
            
            # Clear memory
            del trained_model
            torch.cuda.empty_cache()
            gc.collect()
        
        # Compare all models
        model_evaluator.compare_models(results)
        
        logger.info("Model training and evaluation complete.")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()
