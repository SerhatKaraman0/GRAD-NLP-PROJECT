from src.common_imports import * # noqa: F403, F405
from src.logging_config import *  # noqa: F403, F405
from src.model_builder import ModelBuilder

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    precision_score, recall_score, accuracy_score, 
    mean_squared_error, mean_absolute_error, r2_score
)

class ModelEvaluator:
    """Class for evaluating sentiment analysis models"""
    
    def __init__(self, save_data_dir):
        """
        Initialize the ModelEvaluator
        
        Args:
            save_data_dir (str): Directory to save evaluation results
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.SAVE_DATA_DIR = save_data_dir
        self.metrics_dir = os.path.join(save_data_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        
    def evaluate_model(self, model, X_test, y_test, model_type='ensemble'):
        """
        Evaluate the PyTorch model on test data
        
        Args:
            model (torch.nn.Module): The model to evaluate
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels
            model_type (str): Type of model ('simple', 'deep', 'stacked', or 'ensemble')
            
        Returns:
            tuple: (mse, mae, acc, y_pred) - Metrics and predictions
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        # Convert to PyTorch tensors
        X_test_tensor = torch.tensor(X_test, dtype=torch.long).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
        
        # Create dataloader for batched evaluation
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor.view(-1, 1))
        test_loader = DataLoader(test_dataset, batch_size=256)
        
        # Evaluate
        test_loss = 0.0
        all_preds = []
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for batch_X, batch_y in tqdm(test_loader, desc="Evaluating"):
                outputs = model(batch_X)
                test_loss += criterion(outputs, batch_y).item()
                all_preds.append(outputs.cpu().numpy())
        
        # Combine predictions and convert to numpy
        y_pred = np.vstack(all_preds).flatten()
        
        # Clamp predictions to valid score range (1-5)
        y_pred_clamped = np.clip(y_pred, 1, 5)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred_clamped)
        mse = mean_squared_error(y_test, y_pred_clamped)
        rmse = np.sqrt(mse)
        
        # Calculate accuracy (rounded predictions within 1-5 range)
        y_pred_rounded = np.round(y_pred_clamped)
        acc = accuracy_score(y_test, y_pred_rounded)
        
        # Calculate other metrics
        precision = precision_score(y_test, y_pred_rounded, average='macro')
        recall = recall_score(y_test, y_pred_rounded, average='macro')
        f1 = f1_score(y_test, y_pred_rounded, average='macro')
        r2 = r2_score(y_test, y_pred_clamped)
        
        # Log results
        self.logger.info(f"Model evaluation: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, Accuracy={acc:.4f}")
        self.logger.info(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, R²={r2:.4f}")
        
        # Create visualizations
        self._create_evaluation_visualizations(y_test, y_pred_clamped, mae, model_type)
        
        # Compile results
        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'r2': r2
        }
        
        return mse, mae, acc, y_pred_clamped, results

    def _create_evaluation_visualizations(self, y_test, y_pred, mae, model_type):
        """
        Create and save evaluation visualizations
        
        Args:
            y_test (numpy.ndarray): True labels
            y_pred (numpy.ndarray): Predicted labels
            mae (float): Mean Absolute Error
            model_type (str): Type of model
        """
        # Create a confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(np.round(y_test), np.round(y_pred))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.metrics_dir, f'confusion_matrix_{model_type}.png'))
        plt.close()
        
        # Create a distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(y_test, alpha=0.5, label='True Values')
        plt.hist(y_pred, alpha=0.5, label='Predictions')
        plt.title('Distribution of True vs Predicted Values')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.metrics_dir, f'distribution_plot_{model_type}.png'))
        plt.close()
        
        # Create a scatter plot of true vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.plot([1, 5], [1, 5], 'r--')  # Diagonal line for perfect predictions
        plt.title(f'True vs Predicted Values (MAE: {mae:.4f})')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.grid(True)
        plt.savefig(os.path.join(self.metrics_dir, f'true_vs_pred_{model_type}.png'))
        plt.close()
        
    def compare_models(self, model_results):
        """
        Compare multiple models and save comparison results
        
        Args:
            model_results (dict): Dictionary of model results keyed by model type
            
        Returns:
            pandas.DataFrame: DataFrame with comparison results
        """
        # Convert results to DataFrame
        comparison_data = []
        
        for model_type, results in model_results.items():
            comparison_data.append({
                'Model': model_type,
                'MSE': results.get('mse', 0),
                'RMSE': results.get('rmse', 0),
                'MAE': results.get('mae', 0),
                'Accuracy': results.get('accuracy', 0),
                'Precision': results.get('precision', 0),
                'Recall': results.get('recall', 0),
                'F1': results.get('f1', 0),
                'R²': results.get('r2', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save to CSV
        comparison_path = os.path.join(self.SAVE_DATA_DIR, 'model_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        
        self.logger.info(f"Model comparison saved to {comparison_path}")
        
        return comparison_df
