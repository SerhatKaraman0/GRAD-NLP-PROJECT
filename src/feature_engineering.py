from src.common_imports import *  # noqa: F403, F405
from src.nlpmodel import NlpModel
from src.logging_config import *  # noqa: F403, F405
from utils.helper import CONTRACTIONS_DICT, SLANG_DICT  

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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# To ensure compatibility across environments
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up logging with a more efficient configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineering(NlpModel):
    def __init__(self, batch_size=10_000, max_features=7000):
        """
        Initialize paths and load dataset.
        
        Args:
            batch_size: Size of batches for processing
            max_features: Maximum number of features for TF-IDF vectorizer
        """
        super().__init__()
        self.SAVE_DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.PREPROCESSED_DATA_PATH = os.path.join(self.SAVE_DATA_DIR, "PREPROCESSED_Reviews.csv")
        self.batch_size = batch_size
        self.vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=max_features,
            lowercase=True,
            sublinear_tf=True  # Apply sublinear tf scaling (1 + log(tf))
        )
        
        # Defer dataframe loading to when it's needed
        self.df = None
        self.df_size = None
        self.n_batches = None

    def load_data(self):
        """Load data from file and initialize related properties."""
        logger.info(f"Loading data from {self.PREPROCESSED_DATA_PATH}")
        # Use low_memory=False to avoid mixed type inference warnings
        self.df = pd.read_csv(self.PREPROCESSED_DATA_PATH, low_memory=False)
        self.df_size = len(self.df)
        self.n_batches = (self.df_size + self.batch_size - 1) // self.batch_size
        logger.info(f"Data loaded: {self.df_size} records, {self.n_batches} batches")
        
    def preprocess_text(self):
        """Preprocess text data by handling NaNs."""
        if self.df is None:
            self.load_data()
        logger.info("Preprocessing text data")
        # Fill missing values with empty string (more efficient than fillna)
        self.df['Text'] = self.df['Text'].fillna('', inplace=False)

    def _vectorize_tfidf_batch(self, start_idx, end_idx, fit=False):
        """
        Vectorize a batch of data using TF-IDF.
        
        Args:
            start_idx: Starting index of the batch
            end_idx: Ending index of the batch
            fit: Whether to fit the vectorizer on this batch
            
        Returns:
            tuple: (TF-IDF sparse matrix, updated fit flag)
        """
        batch = self.df.iloc[start_idx:end_idx]['Text']
        
        if fit:
            # Only fit on first batch
            batch_tfidf = self.vectorizer.fit_transform(batch)
            return batch_tfidf, False
        else:
            # For subsequent batches, just transform
            batch_tfidf = self.vectorizer.transform(batch)
            return batch_tfidf, False
    
    def process_in_batches(self):
        """
        Process the data in batches to handle large datasets efficiently.
        
        Returns:
            scipy.sparse.csr_matrix: Combined TF-IDF matrix
        """
        self.preprocess_text()
        sparse_matrices = []
        fit = True  # Flag to fit only the first batch
        
        # Use tqdm for progress tracking
        for start_idx in tqdm(range(0, self.df_size, self.batch_size), 
                              desc="Processing batches", 
                              unit="batch"):
            end_idx = min(start_idx + self.batch_size, self.df_size)
            batch_tfidf, fit = self._vectorize_tfidf_batch(start_idx, end_idx, fit)
            
            # Store the sparse matrix
            sparse_matrices.append(batch_tfidf)
            
            # Clean up memory
            gc.collect()
        
        # Combine all sparse matrices efficiently
        logger.info("Combining sparse matrices")
        all_tfidf_matrix = sparse.vstack(sparse_matrices)
        return all_tfidf_matrix
    
    def _vectorize_tfidf(self):
        """Vectorize the entire dataset using batched processing."""
        logger.info("Starting TF-IDF vectorization")
        return self.process_in_batches()

    def build_model(self, input_dim):
        """
        Build a neural network model for sentiment analysis.
        
        Args:
            input_dim: Dimension of the input features
            
        Returns:
            tf.keras.models.Sequential: Compiled Keras model
        """
        logger.info(f"Building model with input dimension {input_dim}")
        
        # Set up for mixed precision training for better performance on compatible GPUs
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Using mixed precision training")
        except:
            logger.info("Mixed precision not available, using default precision")
        
        # Model architecture (Dense Feedforward Network)
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(512, activation='relu', kernel_initializer='he_normal'),
            Dropout(0.3),
            Dense(256, activation='relu', kernel_initializer='he_normal'),
            Dense(1, activation='linear')  # Regression for rating prediction
        ])
        
        # Use Adam optimizer with learning rate scheduling
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer, 
            loss='mse',  # Mean squared error for regression
            metrics=['mae']  # Mean absolute error
        )
        
        return model

    def train_model(self, X, y):
        """
        Train a model on the TF-IDF matrix and visualize training metrics.
        
        Args:
            X: Training features
            y: Target values
            
        Returns:
            tf.keras.models.Sequential: Trained model
        """
        # Build the model
        model = self.build_model(X.shape[1])
        
        # Set up callbacks for training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                os.path.join(self.SAVE_DATA_DIR, "sentiment_model_best.h5"),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            # Add learning rate reduction on plateau
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Add custom callback to track learning rate only if not using mixed precision
        try:
            # For standard optimizers
            if hasattr(model.optimizer, 'lr'):
                callbacks.append(
                    tf.keras.callbacks.LambdaCallback(
                        on_epoch_end=lambda epoch, logs: logs.update({'lr': float(model.optimizer.lr.numpy())})
                    )
                )
            # For LossScaleOptimizer (mixed precision)
            elif hasattr(model.optimizer, '_optimizer') and hasattr(model.optimizer._optimizer, 'lr'):
                callbacks.append(
                    tf.keras.callbacks.LambdaCallback(
                        on_epoch_end=lambda epoch, logs: logs.update({'lr': float(model.optimizer._optimizer.lr.numpy())})
                    )
                )
            else:
                logger.warning("Could not access learning rate for tracking")
        except Exception as e:
            logger.warning(f"Error setting up learning rate tracking: {e}")
        
        # Train the model with batching for memory efficiency
        logger.info("Starting model training")
        history = model.fit(
            X, y,
            epochs=10,
            batch_size=512,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=2  # Less verbose output
        )
        
        logger.info("Model training completed")
        
        # Visualize training history
        self.visualize_training_history(history)
        
        return model

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the model on the test set and visualize results.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target values
            
        Returns:
            tuple: (loss, mean absolute error, predicted values)
        """
        logger.info("Evaluating model on test data")
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Model evaluation: Loss={loss:.4f}, MAE={mae:.4f}")
        
        # Get predictions
        y_pred = model.predict(X_test, verbose=0).flatten()
        
        # Create metrics directory if it doesn't exist
        metrics_dir = os.path.join(self.SAVE_DATA_DIR, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # 1. Actual vs Predicted scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.xlabel('Actual Ratings')
        plt.ylabel('Predicted Ratings')
        plt.title('Actual vs Predicted Ratings')
        plt.savefig(os.path.join(metrics_dir, 'actual_vs_predicted.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Error distribution
        errors = y_pred - y_test
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.title(f'Error Distribution (MAE: {mae:.4f})')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.savefig(os.path.join(metrics_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Error by rating category
        plt.figure(figsize=(12, 6))
        df_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Error': errors})
        df_results['AbsError'] = np.abs(errors)
        df_results['RatingCategory'] = pd.cut(df_results['Actual'], bins=[0, 1.5, 2.5, 3.5, 4.5, 5.5], 
                                             labels=['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars'])
        
        sns.boxplot(x='RatingCategory', y='AbsError', data=df_results)
        plt.xlabel('Rating Category')
        plt.ylabel('Absolute Error')
        plt.title('Error Distribution by Rating Category')
        plt.savefig(os.path.join(metrics_dir, 'error_by_category.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Save metrics to CSV for further analysis
        metrics_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Error': errors,
            'AbsError': np.abs(errors)
        })
        metrics_df.to_csv(os.path.join(metrics_dir, 'prediction_metrics.csv'), index=False)
        
        # 5. Summary metrics table
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        summary_metrics = {
            'Metric': ['MSE', 'MAE', 'RMSE', 'R²'],
            'Value': [loss, mae, rmse, r2]
        }
        summary_df = pd.DataFrame(summary_metrics)
        summary_df.to_csv(os.path.join(metrics_dir, 'summary_metrics.csv'), index=False)
        
        # Log metrics
        logger.info(f"Summary Metrics:\n{summary_df.to_string()}")
        
        return loss, mae, y_pred

    def visualize_training_history(self, history):
        """
        Visualize the model training history.
        
        Args:
            history: Keras history object from model.fit()
        """
        metrics_dir = os.path.join(self.SAVE_DATA_DIR, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Convert history to DataFrame
        history_df = pd.DataFrame(history.history)
        
        # Save history to CSV
        history_df.to_csv(os.path.join(metrics_dir, 'training_history.csv'), index=False)
        
        # Plot training & validation loss
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history_df['loss'], label='Training Loss')
        plt.plot(history_df['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history_df['mae'], label='Training MAE')
        plt.plot(history_df['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Learning rate plot if available
        if 'lr' in history_df.columns:
            plt.figure(figsize=(10, 4))
            plt.plot(history_df['lr'])
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.savefig(os.path.join(metrics_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Training history visualizations saved to {metrics_dir}")

    def build_interactive_dashboard(self, model, X_test, y_test, y_pred):
        """
        Build an interactive HTML dashboard of model metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            y_pred: Predicted values
        """
        metrics_dir = os.path.join(self.SAVE_DATA_DIR, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        mse = mean_squared_error(y_test, y_pred)
        
        # For classification-like metrics, round predictions
        y_test_rounded = np.round(y_test).astype(int)
        y_pred_rounded = np.round(y_pred).astype(int)
        
        # Cap to valid range
        y_pred_rounded = np.clip(y_pred_rounded, 1, 5)
        
        # Create a confusion matrix
        conf_matrix = confusion_matrix(y_test_rounded, y_pred_rounded)
        
        # Function to generate base64 image from figure
        def fig_to_base64(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            return img_str
        
        # Create figures and convert to base64
        
        # 1. Actual vs Predicted
        fig1 = plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.xlabel('Actual Ratings')
        plt.ylabel('Predicted Ratings')
        plt.title('Actual vs Predicted Ratings')
        plt.grid(True, alpha=0.3)
        actual_vs_pred_img = fig_to_base64(fig1)
        plt.close(fig1)
        
        # 2. Error distribution
        errors = y_pred - y_test
        fig2 = plt.figure(figsize=(8, 6))
        sns.histplot(errors, kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.title(f'Error Distribution (MAE: {mae:.4f})')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.grid(True, alpha=0.3)
        error_dist_img = fig_to_base64(fig2)
        plt.close(fig2)
        
        # 3. Confusion Matrix
        fig3 = plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=range(1, 6), yticklabels=range(1, 6))
        plt.xlabel('Predicted Rating')
        plt.ylabel('Actual Rating')
        plt.title('Confusion Matrix (Rounded Ratings)')
        conf_matrix_img = fig_to_base64(fig3)
        plt.close(fig3)
        
        # Create HTML dashboard
        html_content = f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Sentiment Analysis Model Dashboard</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .header {{
                    background-color: #4a86e8;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .metric-box {{
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    padding: 15px;
                    margin-bottom: 15px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metrics-container {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-between;
                    margin-bottom: 20px;
                }}
                .metric-item {{
                    width: 22%;
                    text-align: center;
                    background-color: #e8f4f8;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #4a86e8;
                }}
                .charts-container {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-between;
                }}
                .chart-box {{
                    width: 48%;
                    margin-bottom: 20px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    padding: 15px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .chart-box img {{
                    width: 100%;
                    height: auto;
                }}
                .full-width {{
                    width: 100%;
                }}
                h2 {{
                    color: #4a86e8;
                }}
                @media (max-width: 768px) {{
                    .metric-item {{
                        width: 48%;
                        margin-bottom: 15px;
                    }}
                    .chart-box {{
                        width: 100%;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Sentiment Analysis Model Dashboard</h1>
                    <p>Model performance metrics and visualizations</p>
                </div>
                
                <div class="metric-box">
                    <h2>Key Performance Metrics</h2>
                    <div class="metrics-container">
                        <div class="metric-item">
                            <h3>MAE</h3>
                            <div class="metric-value">{mae:.4f}</div>
                            <p>Mean Absolute Error</p>
                        </div>
                        <div class="metric-item">
                            <h3>MSE</h3>
                            <div class="metric-value">{mse:.4f}</div>
                            <p>Mean Squared Error</p>
                        </div>
                        <div class="metric-item">
                            <h3>RMSE</h3>
                            <div class="metric-value">{rmse:.4f}</div>
                            <p>Root Mean Squared Error</p>
                        </div>
                        <div class="metric-item">
                            <h3>R²</h3>
                            <div class="metric-value">{r2:.4f}</div>
                            <p>Coefficient of Determination</p>
                        </div>
                    </div>
                </div>
                
                <div class="charts-container">
                    <div class="chart-box">
                        <h2>Actual vs Predicted Ratings</h2>
                        <img src="data:image/png;base64,{actual_vs_pred_img}" alt="Actual vs Predicted Plot">
                    </div>
                    <div class="chart-box">
                        <h2>Error Distribution</h2>
                        <img src="data:image/png;base64,{error_dist_img}" alt="Error Distribution">
                    </div>
                    <div class="chart-box full-width">
                        <h2>Confusion Matrix (Rounded Ratings)</h2>
                        <img src="data:image/png;base64,{conf_matrix_img}" alt="Confusion Matrix">
                    </div>
                </div>
                
                <div class="metric-box">
                    <h2>Model Information</h2>
                    <p><strong>Features:</strong> {model.input_shape[1]} TF-IDF features</p>
                    <p><strong>Architecture:</strong> {model.input_shape[1]} → 512 → 256 → 1</p>
                    <p><strong>Test Set Size:</strong> {len(y_test)} samples</p>
                    <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        '''
        
        # Save the HTML dashboard
        dashboard_path = os.path.join(metrics_dir, 'model_dashboard.html')
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Interactive dashboard saved to {dashboard_path}")
        return dashboard_path

    def save_to_parquet(self, df, output_path):
        """
        Save the processed DataFrame to Parquet with gzip compression.
        
        Args:
            df: DataFrame to save
            output_path: Path to save the file (without extension)
        """
        logger.info(f"Saving DataFrame to {output_path}.parquet.gz")
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to Parquet format with gzip compression
        df.to_parquet(f"{output_path}.parquet.gz", compression="gzip")
        
        logger.info(f"Data saved to {output_path}.parquet.gz")

    def load_and_process_data(self):
        """
        Load and process data, returning train/test splits.
        
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        if self.df is None:
            self.load_data()
        
        # Generate TF-IDF features
        X_sparse = self._vectorize_tfidf()
        y = self.df["Score"].values
        
        logger.info(f"TF-IDF matrix shape: {X_sparse.shape}")
        
        # Split the data into training and test sets
        logger.info("Splitting data into train and test sets")
        X_train, X_test, y_train, y_test = train_test_split(
            X_sparse, y, 
            test_size=0.2, 
            random_state=42,
            stratify=None  # Can use stratify=pd.qcut(y, 5, duplicates='drop') if needed
        )
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Check if we need to convert to dense
        if isinstance(X_train, sparse.spmatrix) and tf.config.list_physical_devices('GPU'):
            logger.info("Converting sparse matrices to dense for GPU training")
            X_train_processed = X_train.toarray()
            X_test_processed = X_test.toarray()
        else:
            # Keep as sparse for CPU training or if already dense
            X_train_processed = X_train
            X_test_processed = X_test
            
        return X_train_processed, y_train, X_test_processed, y_test
    
    def save_feature_importance(self, model, output_path=None):
        """
        Save feature importance scores from the model.
        
        Args:
            model: Trained model
            output_path: Path to save the scores (optional)
        """
        if output_path is None:
            output_path = os.path.join(self.SAVE_DATA_DIR, "feature_importance.csv")
            
        logger.info("Extracting feature importance")
        
        # Get feature names from vectorizer
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get weights from the first layer
        weights = model.layers[0].get_weights()[0]
        
        # Create a DataFrame with feature names and weights
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(weights.mean(axis=1))
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Save to CSV
        importance_df.to_csv(output_path, index=False)
        logger.info(f"Feature importance saved to {output_path}")
        
        return importance_df.head(20)  # Return top 20 features for inspection


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
        
        # Train the model
        trained_model = model.train_model(X_train, y_train)
        
        # Evaluate the model with comprehensive metrics
        loss, mae, y_pred = model.evaluate_model(trained_model, X_test, y_test)

        
        # Create interactive dashboard
        dashboard_path = model.build_interactive_dashboard(
            trained_model, X_test, y_test, y_pred
        )
        logger.info(f"Interactive dashboard created at: {dashboard_path}")
        
        # Save the model
        model_path = os.path.join(model.SAVE_DATA_DIR, "sentiment_model_final.h5")
        trained_model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save feature importance
        top_features = model.save_feature_importance(trained_model)
        logger.info(f"Top features:\n{top_features}")
        
    except Exception as e:
        logger.error(f"Error in processing: {e}", exc_info=True)
        raise



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
        
        # Train the model
        trained_model = model.train_model(X_train, y_train)
        
        # Evaluate the model with comprehensive metrics
        loss, mae, y_pred = model.evaluate_model(trained_model, X_test, y_test)
        
        # Create interactive dashboard
        dashboard_path = model.build_interactive_dashboard(
            trained_model, X_test, y_test, y_pred
        )
        logger.info(f"Interactive dashboard created at: {dashboard_path}")
        
        # Save the model
        model_path = os.path.join(model.SAVE_DATA_DIR, "sentiment_model_final.h5")
        trained_model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save feature importance
        top_features = model.save_feature_importance(trained_model)
        logger.info(f"Top features:\n{top_features}")
        
    except Exception as e:
        logger.error(f"Error in processing: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
