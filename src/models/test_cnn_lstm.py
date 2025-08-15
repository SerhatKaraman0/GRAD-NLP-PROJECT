#!/usr/bin/env python
# filepath: /Users/user/Desktop/Projects/NLP-Learning/src/models/test_cnn_lstm.py

"""
Test script for the CNNBiLSTMClassifier model.
This script evaluates the trained model on test data and generates predictions.
"""

import os
import sys
import torch
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Import project modules
from src.core.logging_config import setup_logging, get_logger
from src.features.embeddings import EmbeddingProcessor
from src.models.models import CNNBiLSTMClassifier

# Setup logger
setup_logging()
logger = get_logger('CNNBiLSTM_Evaluation')

def load_model(model_path, config):
    """
    Load trained model from path
    
    Args:
        model_path (str): Path to the model file
        config (dict): Model configuration parameters
        
    Returns:
        model: The loaded model
    """
    try:
        # Initialize embedding processor to get vocab size and embedding matrix
        embedding_processor = EmbeddingProcessor(
            batch_size=config['batch_size'],
            max_features=config['max_features'],
            embedding_dim=config['embedding_dim']
        )
        
        # Get a dummy text sequence to initialize embedding processor
        dummy_sequences, embedding_matrix = embedding_processor.prepare_embeddings(["dummy text"])
        
        # Initialize model
        model = CNNBiLSTMClassifier(
            input_size=config['max_seq_length'],
            embedding_dim=config['embedding_dim'],
            vocab_size=embedding_processor.max_features + 1,
            num_classes=config['num_classes']
        )
        
        # Load state dictionary
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        logger.info(f"Model loaded from {model_path}")
        
        # Set to evaluation mode
        model.eval()
        return model, embedding_processor
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def prepare_test_data(data_path, embedding_processor, test_size=0.2, random_state=42):
    """
    Prepare test data for evaluation
    
    Args:
        data_path (str): Path to the data file
        embedding_processor: Initialized embedding processor
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed
        
    Returns:
        tuple: Test data and labels
    """
    try:
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} records from {data_path}")
        
        # Check required columns
        if 'Text' not in df.columns or 'Score' not in df.columns:
            raise ValueError("Data must contain 'Text' and 'Score' columns")
        
        # Split data to get test set
        _, X_test, _, y_test = train_test_split(
            df['Text'].tolist(), 
            df['Score'].values - 1,  # Convert 1-5 scale to 0-4 for classification
            test_size=test_size,
            random_state=random_state,
            stratify=df['Score']
        )
        
        # Convert text to sequences
        X_sequences, _ = embedding_processor.prepare_embeddings(X_test)
        
        logger.info(f"Test data prepared: {len(X_sequences)} samples")
        return X_sequences, y_test
        
    except Exception as e:
        logger.error(f"Failed to prepare test data: {e}")
        raise

def evaluate_model(model, X_test, y_test, device, batch_size=64):
    """
    Evaluate model on test data
    
    Args:
        model: The trained model
        X_test: Test features
        y_test: Test labels
        device: Computation device (CPU/GPU)
        batch_size (int): Batch size for evaluation
        
    Returns:
        dict: Evaluation metrics
    """
    try:
        # Move model to device
        model = model.to(device)
        
        # Lists to store predictions and actual labels
        all_preds = []
        all_probs = []
        
        # Process data in batches
        num_samples = len(X_test)
        for i in range(0, num_samples, batch_size):
            # Get batch
            batch_X = X_test[i:min(i + batch_size, num_samples)]
            
            # Convert to tensor
            inputs = torch.tensor(batch_X, dtype=torch.long).to(device)
            
            # Make predictions
            with torch.no_grad():
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
            
            # Store predictions and probabilities
            all_preds.extend(predictions.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, all_preds)
        cm = confusion_matrix(y_test, all_preds)
        report = classification_report(y_test, all_preds, output_dict=True)
        rmse = np.sqrt(mean_squared_error(y_test, all_preds))
        
        # Prepare results
        results = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'rmse': rmse,
            'predictions': all_preds,
            'probabilities': all_probs,
            'true_labels': y_test
        }
        
        logger.info(f"Evaluation completed with accuracy: {accuracy:.4f}, RMSE: {rmse:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

def plot_results(results, save_path=None):
    """
    Plot evaluation results
    
    Args:
        results (dict): Evaluation results
        save_path (str): Path to save the plots
    """
    try:
        # Create directory if needed
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot confusion matrix
        cm = results['confusion_matrix']
        cax = ax1.matshow(cm, cmap='Blues')
        fig.colorbar(cax, ax=ax1)
        
        # Set labels for confusion matrix
        ax1.set_title('Confusion Matrix', fontsize=14)
        ax1.set_xlabel('Predicted Label (0-4)', fontsize=12)
        ax1.set_ylabel('True Label (0-4)', fontsize=12)
        
        # Add text annotations to confusion matrix
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax1.text(j, i, str(cm[i, j]), ha='center', va='center', 
                         color='black' if cm[i, j] < cm.max() / 2 else 'white')
        
        # Plot prediction distribution
        ax2.hist(results['predictions'], bins=range(6), alpha=0.7, label='Predicted')
        ax2.hist(results['true_labels'], bins=range(6), alpha=0.7, label='Actual')
        ax2.set_title('Distribution of Predictions vs Actual Labels', fontsize=14)
        ax2.set_xlabel('Rating (0-4)', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plots saved to {save_path}")
            
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting results: {e}")

def analyze_embedding_performance(results):
    """
    Analyze the effectiveness of the embeddings based on prediction results
    
    Args:
        results (dict): Evaluation results
    
    Returns:
        dict: Analysis of embedding effectiveness
    """
    try:
        # Get predictions and true labels
        preds = results['predictions']
        true_labels = results['true_labels']
        
        # Calculate error by magnitude (distance between true and predicted)
        error_magnitude = np.abs(preds - true_labels)
        
        # Analysis metrics
        analysis = {
            'avg_error_magnitude': np.mean(error_magnitude),
            'max_error_magnitude': np.max(error_magnitude),
            'perfect_predictions': np.sum(error_magnitude == 0) / len(error_magnitude),
            'within_one_class': np.sum(error_magnitude <= 1) / len(error_magnitude),
            'large_errors': np.sum(error_magnitude >= 2) / len(error_magnitude),
        }
        
        logger.info(f"Embedding Performance Analysis:")
        logger.info(f"- Perfect predictions: {analysis['perfect_predictions']:.2%}")
        logger.info(f"- Predictions within one class: {analysis['within_one_class']:.2%}")
        logger.info(f"- Average error magnitude: {analysis['avg_error_magnitude']:.4f}")
        logger.info(f"- Large errors (≥2 classes): {analysis['large_errors']:.2%}")
        
        return analysis
    except Exception as e:
        logger.error(f"Error during embedding analysis: {e}")
        return {}

def main(args):
    """
    Main function for model evaluation
    
    Args:
        args: Command line arguments
    """
    try:
        # Define default configuration
        config = {
            'batch_size': args.batch_size,
            'embedding_dim': args.embedding_dim,
            'max_features': 20000,
            'max_seq_length': 500,
            'num_classes': 5
        }
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load model
        model, embedding_processor = load_model(args.model_path, config)
        
        # Prepare test data
        X_test, y_test = prepare_test_data(
            args.data_path,
            embedding_processor,
            test_size=args.test_size,
            random_state=args.seed
        )
        
        # Evaluate model
        results = evaluate_model(model, X_test, y_test, device, batch_size=config['batch_size'])
        
        # Display results
        print(f"\nEvaluation Results:")
        print(f"===================")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"RMSE: {results['rmse']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, results['predictions']))
        
        # Analyze embedding performance
        embedding_analysis = analyze_embedding_performance(results)
        
        # Plot results if requested
        if not args.no_plot:
            save_path = os.path.join(project_root, 'data', 'metrics', 'cnn_bilstm_evaluation.png') if not args.no_save else None
            plot_results(results, save_path)
            
        # Save results if requested
        if not args.no_save:
            # Include embedding dimension in filenames
            metrics_path = os.path.join(project_root, 'data', 'metrics', f'cnn_bilstm_{args.embedding_dim}d_metrics.csv')
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            
            # Prepare DataFrame for metrics
            metrics_df = pd.DataFrame({
                'embedding_dim': [args.embedding_dim],
                'accuracy': [results['accuracy']],
                'rmse': [results['rmse']],
                'f1_macro': [results['classification_report']['macro avg']['f1-score']],
                'f1_weighted': [results['classification_report']['weighted avg']['f1-score']]
            })
            
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Metrics saved to {metrics_path}")
            
            # Save predictions
            preds_path = os.path.join(project_root, 'data', 'metrics', f'cnn_bilstm_{args.embedding_dim}d_predictions.csv')
            preds_df = pd.DataFrame({
                'embedding_dim': args.embedding_dim,
                'true_label': results['true_labels'],
                'predicted_label': results['predictions']
            })
            
            preds_df.to_csv(preds_path, index=False)
            logger.info(f"Predictions saved to {preds_path}")
            
            # Save embedding analysis
            analysis_path = os.path.join(project_root, 'data', 'metrics', f'cnn_bilstm_{args.embedding_dim}d_embedding_analysis.csv')
            analysis_df = pd.DataFrame([embedding_analysis])
            analysis_df['embedding_dim'] = args.embedding_dim
            analysis_df.to_csv(analysis_path, index=False)
            logger.info(f"Embedding analysis saved to {analysis_path}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate CNN BiLSTM model')
    
    parser.add_argument('--model_path', type=str, 
                        default=os.path.join(project_root, 'data', 'models', 'best_model_cnn_bilstm.pth'),
                        help='Path to the trained model')
                        
    parser.add_argument('--data_path', type=str, 
                        default=os.path.join(project_root, 'data', 'PREPROCESSED_Reviews.csv'),
                        help='Path to the test data')
                        
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
                        
    parser.add_argument('--embedding_dim', type=int, default=100,
                        help='Embedding dimension used in the model')
                        
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
                        
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
                        
    parser.add_argument('--cpu', action='store_true',
                        help='Force using CPU even if GPU is available')
                        
    parser.add_argument('--no_plot', action='store_true',
                        help='Disable result plotting')
                        
    parser.add_argument('--no_save', action='store_true',
                        help='Disable saving results to files')
    
    args = parser.parse_args()
    main(args)
