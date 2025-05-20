#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module provides functionality for comparing different NLP models.
It loads metrics from different model runs and generates comparative visualizations.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union
import logging

class ModelComparison:
    """Class for comparing different models based on their metrics"""
    
    def __init__(self, save_data_dir: str):
        """
        Initialize the ModelComparison
        
        Args:
            save_data_dir (str): Directory where model metrics are saved
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.SAVE_DATA_DIR = save_data_dir
        self.metrics_dir = os.path.join(save_data_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        
    def load_model_metrics(self, model_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load metrics from different model types
        
        Args:
            model_types (List[str], optional): List of model types to load. 
                                              If None, loads all available models.
                                              
        Returns:
            pd.DataFrame: DataFrame containing metrics for all models
        """
        if model_types is None:
            model_types = ['simple', 'deep', 'stacked', 'ensemble']
            
        # Check which models have metrics available
        available_models = []
        for model_type in model_types:
            metrics_path = os.path.join(self.metrics_dir, f'prediction_metrics_{model_type}.csv')
            if os.path.exists(metrics_path):
                available_models.append(model_type)
            else:
                self.logger.warning(f"Metrics for model type '{model_type}' not found at {metrics_path}")
        
        if not available_models:
            self.logger.warning("No model metrics found. Returning empty DataFrame.")
            return pd.DataFrame()
        
        # Load metrics for available models
        metrics_list = []
        for model_type in available_models:
            try:
                # Try to load from CSV
                metrics_path = os.path.join(self.metrics_dir, f'prediction_metrics_{model_type}.csv')
                if os.path.exists(metrics_path):
                    metrics_df = pd.read_csv(metrics_path)
                    metrics_df['model_type'] = model_type
                    metrics_list.append(metrics_df)
                else:
                    # Try to load from JSON if CSV doesn't exist
                    history_path = os.path.join(self.SAVE_DATA_DIR, f'training_history_{model_type}.json')
                    if os.path.exists(history_path):
                        with open(history_path, 'r') as f:
                            history = json.load(f)
                        
                        # Extract last epoch metrics
                        if isinstance(history, dict) and 'val_loss' in history:
                            last_epoch = {
                                'model_type': model_type,
                                'val_loss': history['val_loss'][-1],
                                'val_accuracy': history.get('val_accuracy', [0])[-1],
                                'val_mae': history.get('val_mae', [0])[-1],
                                'val_mse': history.get('val_mse', [0])[-1]
                            }
                            metrics_list.append(pd.DataFrame([last_epoch]))
            except Exception as e:
                self.logger.error(f"Error loading metrics for model type '{model_type}': {str(e)}")
        
        if not metrics_list:
            self.logger.warning("Failed to load any model metrics. Returning empty DataFrame.")
            return pd.DataFrame()
        
        # Combine all metrics
        combined_metrics = pd.concat(metrics_list, ignore_index=True)
        return combined_metrics
    
    def create_comparison_table(self, metrics_df: pd.DataFrame) -> str:
        """
        Create an HTML table comparing different models
        
        Args:
            metrics_df (pd.DataFrame): DataFrame containing model metrics
            
        Returns:
            str: HTML formatted table
        """
        if metrics_df.empty:
            return "<p>No model metrics available for comparison.</p>"
        
        # Define metrics to include in comparison
        comparison_metrics = [
            'accuracy', 'precision', 'recall', 'f1', 'mse', 'mae', 'rmse', 'r2'
        ]
        
        # Calculate RMSE if not present
        if 'mse' in metrics_df.columns and 'rmse' not in metrics_df.columns:
            metrics_df['rmse'] = np.sqrt(metrics_df['mse'])
        
        # Select subset of columns that are available
        available_metrics = [col for col in comparison_metrics if col in metrics_df.columns]
        if not available_metrics:
            return "<p>No comparable metrics found in the dataset.</p>"
        
        # Create a pivot table for comparison
        try:
            comparison_df = metrics_df.pivot_table(
                index='model_type', 
                values=available_metrics, 
                aggfunc='mean'  # In case there are multiple rows per model
            )
            
            # Format the table for HTML
            formatted_df = comparison_df.copy()
            for col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")
            
            # Bold the best value in each column
            for col in comparison_df.columns:
                # For metrics where lower is better (mse, mae, rmse)
                if col in ['mse', 'mae', 'rmse']:
                    best_idx = comparison_df[col].idxmin()
                else:  # For metrics where higher is better (accuracy, precision, recall, f1, r2)
                    best_idx = comparison_df[col].idxmax()
                
                formatted_df.loc[best_idx, col] = f"<strong>{formatted_df.loc[best_idx, col]}</strong>"
            
            # Convert to HTML
            html_table = formatted_df.to_html(escape=False)
            
            # Add CSS styling
            styled_table = f"""
            <div class="table-responsive">
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Model Type</th>
                            {' '.join([f'<th>{m.upper()}</th>' for m in available_metrics])}
                        </tr>
                    </thead>
                    <tbody>
                        {' '.join([f'<tr><td>{row[0]}</td>{" ".join([f"<td>{val}</td>" for val in row[1:]])}</tr>' 
                                  for row in formatted_df.reset_index().values.tolist()])}
                    </tbody>
                </table>
            </div>
            """
            
            return styled_table
            
        except Exception as e:
            self.logger.error(f"Error creating comparison table: {str(e)}")
            return f"<p>Error creating model comparison table: {str(e)}</p>"
    
    def create_comparison_visualizations(self) -> Dict[str, str]:
        """
        Create visualizations comparing different models
        
        Returns:
            Dict[str, str]: Dictionary with paths to saved visualizations
        """
        # Load model metrics
        metrics_df = self.load_model_metrics()
        
        if metrics_df.empty:
            self.logger.warning("No metrics available to create comparison visualizations")
            return {}
        
        visualizations = {}
        
        # Bar chart comparing key metrics
        try:
            self.logger.info("Creating bar chart comparison of key metrics")
            
            # Get metrics to compare (exclude model_type and any non-numeric columns)
            numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns.tolist()
            metrics_to_compare = [col for col in numeric_cols 
                                if col in ['accuracy', 'mse', 'mae', 'f1', 'precision', 'recall']]
            
            if metrics_to_compare:
                # Melt the DataFrame for easier plotting
                plot_df = pd.melt(
                    metrics_df, 
                    id_vars=['model_type'], 
                    value_vars=metrics_to_compare, 
                    var_name='metric', 
                    value_name='value'
                )
                
                # Create subplot for each metric
                fig, axes = plt.subplots(
                    len(metrics_to_compare), 1, 
                    figsize=(10, 4 * len(metrics_to_compare)),
                    sharex=True
                )
                
                # If only one metric, axes isn't array
                if len(metrics_to_compare) == 1:
                    axes = [axes]
                
                for i, metric in enumerate(metrics_to_compare):
                    metric_df = plot_df[plot_df['metric'] == metric]
                    sns.barplot(
                        x='model_type', 
                        y='value', 
                        data=metric_df,
                        palette='viridis',
                        ax=axes[i]
                    )
                    axes[i].set_title(f'{metric.upper()}')
                    axes[i].set_ylabel(metric)
                    
                    # Add values on top of bars
                    for j, bar in enumerate(axes[i].patches):
                        value = bar.get_height()
                        axes[i].text(
                            bar.get_x() + bar.get_width()/2., 
                            value + 0.01,
                            f'{value:.4f}',
                            ha='center'
                        )
                
                plt.xlabel('Model Type')
                plt.tight_layout()
                
                # Save the figure
                output_path = os.path.join(self.metrics_dir, 'model_comparison_metrics.png')
                plt.savefig(output_path)
                plt.close()
                
                visualizations['model_comparison_metrics'] = output_path
                self.logger.info(f"Saved model comparison metrics visualization to {output_path}")
        
        except Exception as e:
            self.logger.error(f"Error creating metrics comparison visualization: {str(e)}")
        
        # Create comparison of training history if available
        try:
            self.logger.info("Creating training history comparison")
            
            model_types = metrics_df['model_type'].unique()
            history_data = {}
            
            # Load training history for each model
            for model_type in model_types:
                history_path = os.path.join(self.SAVE_DATA_DIR, f'training_history_{model_type}.json')
                if os.path.exists(history_path):
                    with open(history_path, 'r') as f:
                        history = json.load(f)
                    history_data[model_type] = history
            
            if history_data:
                # Plot training and validation loss
                plt.figure(figsize=(12, 6))
                
                for model_type, history in history_data.items():
                    if 'loss' in history and 'val_loss' in history:
                        epochs = range(1, len(history['loss']) + 1)
                        plt.plot(epochs, history['loss'], marker='o', linestyle='-', alpha=0.7, 
                                label=f'{model_type} - Training Loss')
                        plt.plot(epochs, history['val_loss'], marker='s', linestyle='--', alpha=0.7, 
                                label=f'{model_type} - Validation Loss')
                
                plt.title('Training and Validation Loss Comparison')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save the figure
                output_path = os.path.join(self.metrics_dir, 'model_comparison_history.png')
                plt.savefig(output_path)
                plt.close()
                
                visualizations['model_comparison_history'] = output_path
                self.logger.info(f"Saved model comparison history visualization to {output_path}")
        
        except Exception as e:
            self.logger.error(f"Error creating training history comparison visualization: {str(e)}")
        
        return visualizations
