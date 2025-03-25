from src.common_imports import *  # noqa: F403, F405
from src.nlpmodel import NlpModel
from src.logging_config import *  # noqa: F403, F405
from utils.helper import CONTRACTIONS_DICT, SLANG_DICT
from src.models import SimpleLSTMModel, DeepLSTMModel, StackedLSTMModel, EnsembleModel

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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineering(NlpModel):
    def __init__(self, batch_size=10_000, max_features=7000):
        """Initialize paths and load dataset."""
        super().__init__()
        self.SAVE_DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.PREPROCESSED_DATA_PATH = os.path.join(self.SAVE_DATA_DIR, "PREPROCESSED_Reviews.csv")
        self.batch_size = batch_size
        self.vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=max_features,
            lowercase=True,
            sublinear_tf=True
        )
        self.df = None
        self.df_size = None
        self.n_batches = None
        
    def build_model(self, input_dim):
        """
        Select and build a model for sentiment analysis.
        
        Args:
            input_dim: Dimension of the input features
            
        Returns:
            Model: Selected model instance
        """
        # Create ensemble model by default
        model = EnsembleModel(input_dim)
        return model.build()

    def train_model(self, X, y, model_type='ensemble'):
        """
        Train the specified model type.
        
        Args:
            X: Training features
            y: Target values
            model_type: Type of model to train ('simple', 'deep', 'stacked', or 'ensemble')
            
        Returns:
            tf.keras.models.Model: Trained model
        """
        input_dim = X.shape[1]
        
        # Select model type
        if model_type == 'simple':
            model_class = SimpleLSTMModel(input_dim)
        elif model_type == 'deep':
            model_class = DeepLSTMModel(input_dim)
        elif model_type == 'stacked':
            model_class = StackedLSTMModel(input_dim)
        else:  # default to ensemble
            model_class = EnsembleModel(input_dim)
        
        # Build model
        model = model_class.build()
        
        # Train model
        logger.info(f"Training {model_type} model")
        if model_type == 'ensemble':
            # Prepare inputs for ensemble model
            X_inputs = [X, X, X]
            history = model.fit(
                X_inputs, y,
                epochs=20,
                batch_size=256,
                validation_split=0.2,
                callbacks=model_class.get_callbacks(),
                verbose=1
            )
        else:
            history = model.fit(
                X, y,
                epochs=20,
                batch_size=256,
                validation_split=0.2,
                callbacks=model_class.get_callbacks(),
                verbose=1
            )
        
        logger.info("Model training completed")
        self.visualize_training_history(history)
        
        return model

    def evaluate_model(self, model, X_test, y_test, model_type='ensemble'):
        """
        Evaluate the model on test data.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target values
            model_type: Type of model being evaluated
            
        Returns:
            tuple: (loss, mean absolute error, predicted values)
        """
        logger.info("Evaluating model on test data")
        
        # Prepare inputs based on model type
        if model_type == 'ensemble':
            X_test_input = [X_test, X_test, X_test]
        else:
            X_test_input = X_test
        
        # Evaluate model
        metrics = model.evaluate(X_test_input, y_test, verbose=0)
        loss, mae, acc = metrics
        logger.info(f"Model evaluation: Loss={loss:.4f}, MAE={mae:.4f}, Accuracy={acc:.4f}")
        
        # Get predictions
        y_pred = model.predict(X_test_input, verbose=0).flatten()
        
        # Create visualizations and save metrics
        self._create_evaluation_visualizations(y_test, y_pred, mae)
        
        return loss, mae, y_pred

    def _create_evaluation_visualizations(self, y_test, y_pred, mae):
        """Create and save evaluation visualizations."""
        metrics_dir = os.path.join(self.SAVE_DATA_DIR, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Create visualizations...
        # [Previous visualization code remains the same]

def main():
    """Main execution function."""
    # Enable memory growth for GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU(s) detected: {len(gpus)}")
        except Exception as e:
            logger.warning(f"Error setting GPU memory growth: {e}")
    
    # Initialize FeatureEngineering class
    model = FeatureEngineering(batch_size=10_000, max_features=7000)
    
    try:
        # Load data and process TF-IDF
        X_train, y_train, X_test, y_test = model.load_and_process_data()
        
        # Train and evaluate each model type separately
        model_types = ['simple', 'deep', 'stacked', 'ensemble']
        
        for model_type in model_types:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_type.upper()} LSTM Model")
            logger.info(f"{'='*50}")
            
            # Train the model
            trained_model = model.train_model(X_train, y_train, model_type=model_type)
            
            # Evaluate the model
            loss, mae, y_pred = model.evaluate_model(trained_model, X_test, y_test, model_type=model_type)
            
            # Save the model
            model_path = os.path.join(model.SAVE_DATA_DIR, f"sentiment_model_{model_type}.h5")
            trained_model.save(model_path)
            logger.info(f"{model_type.upper()} model saved to {model_path}")
            
            # Create interactive dashboard for this model
            dashboard_path = model.build_interactive_dashboard(
                trained_model, X_test, y_test, y_pred,
                suffix=f"_{model_type}"
            )
            logger.info(f"Interactive dashboard for {model_type} model created at: {dashboard_path}")
            
            # Clear memory
            del trained_model
            gc.collect()
            if gpus:
                tf.keras.backend.clear_session()
            
        logger.info("\nAll models have been trained and evaluated successfully!")
        
    except Exception as e:
        logger.error(f"Error in processing: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
