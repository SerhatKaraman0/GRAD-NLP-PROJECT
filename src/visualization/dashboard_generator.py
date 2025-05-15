from src.core.common_imports import * # noqa: F403, F405
from src.core.logging_config import *  # noqa: F403, F405

import os
import logging
import datetime
import base64
import pandas as pd
import numpy as np

class DashboardGenerator:
    """Class for generating HTML dashboards for model results"""
    
    def __init__(self, save_data_dir, max_features=10000, embedding_dim=100):
        """
        Initialize the DashboardGenerator
        
        Args:
            save_data_dir (str): Directory where model metrics are saved
            max_features (int): Maximum features used in training
            embedding_dim (int): Embedding dimension used in training
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.SAVE_DATA_DIR = save_data_dir
        self.metrics_dir = os.path.join(save_data_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.max_features = max_features
        self.embedding_dim = embedding_dim
    
    def generate_model_dashboard(self, model_results, model_type='ensemble'):
        """
        Generate an HTML dashboard for model metrics and visualizations
        
        Args:
            model_results (dict): Dictionary containing model evaluation metrics
            model_type (str): Type of model ('simple', 'deep', 'stacked', or 'ensemble')
            
        Returns:
            str: Path to the generated dashboard
        """
        # Extract metrics
        mse = model_results.get('mse', 0)
        mae = model_results.get('mae', 0)
        accuracy = model_results.get('accuracy', 0)
        precision = model_results.get('precision', 0)
        recall = model_results.get('recall', 0)
        f1 = model_results.get('f1', 0)
        rmse = np.sqrt(mse) if mse else 0
        r2 = model_results.get('r2', float('nan'))
        
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Convert images to base64 for embedding in HTML
        confusion_matrix_b64 = self.image_to_base64(os.path.join(self.metrics_dir, f'confusion_matrix_{model_type}.png'))
        true_vs_pred_b64 = self.image_to_base64(os.path.join(self.metrics_dir, f'true_vs_pred_{model_type}.png'))
        distribution_plot_b64 = self.image_to_base64(os.path.join(self.metrics_dir, f'distribution_plot_{model_type}.png'))
        training_history_b64 = self.image_to_base64(os.path.join(self.metrics_dir, f'training_history_{model_type}.png'))
        
        # HTML template
        html_content = f'''<!DOCTYPE html>
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
                        <p>Model performance metrics and visualizations for {model_type.upper()} model</p>
                    </div>
                    
                    <div class="metric-box">
                        <h2>Key Performance Metrics</h2>
                        <div class="metrics-container">
                            <div class="metric-item">
                                <h3>Accuracy</h3>
                                <div class="metric-value">{accuracy:.4f}</div>
                                <p>Classification Accuracy</p>
                            </div>
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
                                <div class="metric-value">{r2}</div>
                                <p>Coefficient of Determination</p>
                            </div>
                            <div class="metric-item">
                                <h3>Precision</h3>
                                <div class="metric-value">{precision:.4f}</div>
                                <p>Precision Score</p>
                            </div>
                            <div class="metric-item">
                                <h3>Recall</h3>
                                <div class="metric-value">{recall:.4f}</div>
                                <p>Recall Score</p>
                            </div>
                            <div class="metric-item">
                                <h3>F1</h3>
                                <div class="metric-value">{f1:.4f}</div>
                                <p>F1 Score</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="charts-container">
                        <div class="chart-box">
                            <h2>Actual vs Predicted Ratings</h2>
                            <img src="data:image/png;base64,{true_vs_pred_b64}">
                        </div>
                        <div class="chart-box">
                            <h2>Error Distribution</h2>
                            <img src="data:image/png;base64,{distribution_plot_b64}">
                        </div>
                        <div class="chart-box full-width">
                            <h2>Confusion Matrix (Rounded Ratings)</h2>
                            <img src="data:image/png;base64,{confusion_matrix_b64}">
                        </div>
                        <div class="chart-box full-width">
                            <h2>Training History</h2>
                            <img src="data:image/png;base64,{training_history_b64}">
                        </div>
                    </div>
                    
                    <div class="metric-box">
                        <h2>Model Information</h2>
                        <p><strong>Features:</strong> {self.max_features} embedding features</p>
                        <p><strong>Architecture:</strong> {model_type.upper()} LSTM model</p>
                        <p><strong>Embedding Dimension:</strong> {self.embedding_dim}</p>
                        <p><strong>Generated:</strong> {timestamp}</p>
                    </div>
                </div>
            </body>
            </html>'''
        
        # Write HTML to file
        dashboard_path = os.path.join(self.metrics_dir, f'model_dashboard_{model_type}.html')
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Model dashboard generated at {dashboard_path}")
        
        # Generate a combined dashboard for all models if this is the last model
        if model_type == 'ensemble':
            self.generate_combined_dashboard()
        
        return dashboard_path
    
    def image_to_base64(self, image_path):
        """
        Convert an image to base64 encoding
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Base64 encoded image
        """
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        return ""
    
    def generate_combined_dashboard(self):
        """
        Generate a combined dashboard comparing all model types
        
        Returns:
            str: Path to the generated combined dashboard
        """
        # Read comparison results
        comparison_path = os.path.join(self.SAVE_DATA_DIR, 'model_comparison.csv')
        if not os.path.exists(comparison_path):
            self.logger.warning(f"Model comparison file not found at {comparison_path}")
            return
        
        # Load comparison data
        try:
            comparison_df = pd.read_csv(comparison_path)
            # Convert DataFrame to HTML table
            comparison_table = comparison_df.to_html(classes='comparison-table', border=0)
        except Exception as e:
            self.logger.error(f"Error loading comparison data: {e}")
            comparison_table = "<p>Error loading comparison data</p>"
        
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # HTML for combined dashboard
        html_content = f'''<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Combined Model Comparison Dashboard</title>
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
                    .comparison-box {{
                        background-color: #f9f9f9;
                        border-radius: 5px;
                        padding: 15px;
                        margin-bottom: 15px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        overflow-x: auto;
                    }}
                    .model-links {{
                        display: flex;
                        justify-content: space-around;
                        margin: 20px 0;
                    }}
                    .model-link {{
                        display: inline-block;
                        padding: 10px 15px;
                        background-color: #4a86e8;
                        color: white;
                        text-decoration: none;
                        border-radius: 5px;
                        font-weight: bold;
                    }}
                    .comparison-table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 15px 0;
                    }}
                    .comparison-table th, .comparison-table td {{
                        padding: 10px;
                        text-align: center;
                        border-bottom: 1px solid #ddd;
                    }}
                    .comparison-table th {{
                        background-color: #e8f4f8;
                        color: #333;
                    }}
                    .comparison-table tr:hover {{
                        background-color: #f5f5f5;
                    }}
                    .footer {{
                        margin-top: 20px;
                        text-align: center;
                        color: #666;
                        font-size: 0.9em;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Sentiment Analysis Model Comparison</h1>
                        <p>Comparative analysis of all trained models</p>
                    </div>
                    
                    <div class="comparison-box">
                        <h2>Model Performance Comparison</h2>
                        {comparison_table}
                    </div>
                    
                    <div class="comparison-box">
                        <h2>Individual Model Dashboards</h2>
                        <p>Click on a model type to view its detailed dashboard:</p>
                        <div class="model-links">
                            <a href="model_dashboard_simple.html" class="model-link">Simple LSTM</a>
                            <a href="model_dashboard_deep.html" class="model-link">Deep LSTM</a>
                            <a href="model_dashboard_stacked.html" class="model-link">Stacked LSTM</a>
                            <a href="model_dashboard_ensemble.html" class="model-link">Ensemble</a>
                        </div>
                    </div>
                    
                    <div class="footer">
                        <p>Generated on: {timestamp}</p>
                    </div>
                </div>
            </body>
            </html>'''
        
        # Write HTML to file
        dashboard_path = os.path.join(self.metrics_dir, 'model_dashboard.html')
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Combined model dashboard generated at {dashboard_path}")
        
        return dashboard_path
