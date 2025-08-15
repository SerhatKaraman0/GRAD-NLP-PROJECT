#!/usr/bin/env python
# filepath: /Users/user/Desktop/Projects/NLP-Learning/run_cnn_bilstm_training.py

"""
Wrapper script to run the CNN BiLSTM model training.
This script sets up configuration and starts the training process.
"""

import os
import sys
import argparse
import json

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the training module
from src.models.train_cnn_lstm import CNNBiLSTMTrainer, main as train_main
from src.features.download_embeddings import ensure_glove_embeddings

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train the CNN BiLSTM classifier model')
    
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=100,
                        help='Embedding dimension (50, 100, 200 for GloVe)')
    parser.add_argument('--freeze_embeddings', action='store_true',
                        help='Freeze the pre-trained embedding layer')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON configuration file')
    
    return parser.parse_args()

def main():
    """Main function to parse arguments and run training"""
    args = parse_arguments()
    
    # Ensure GloVe embeddings are available
    print("Checking for GloVe embeddings...")
    if not ensure_glove_embeddings():
        print("ERROR: GloVe embeddings could not be setup. Exiting.")
        sys.exit(1)
    print("GloVe embeddings are available.")
    
    # Create configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'embedding_dim': args.embedding_dim,
        'freeze_embeddings': args.freeze_embeddings,
        'early_stopping_patience': args.patience,
    }
    
    # Load config from file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
            print(f"Loaded configuration from {args.config}")
    
    # Print configuration summary
    print("\nTraining Configuration:")
    print("=" * 50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 50)
    
    # Create trainer with config
    trainer = CNNBiLSTMTrainer(config)
    
    # Start training
    print("\nStarting training process...")
    print("Using pre-trained GloVe embeddings with dimension:", config['embedding_dim'])
    if config.get('freeze_embeddings', False):
        print("Embeddings will be frozen (not updated during training)")
    else:
        print("Embeddings will be fine-tuned during training")
    
    model, history = trainer.train()
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {history['best_val_loss']:.4f} at epoch {history['best_epoch']}")
    
    # Save model paths
    best_model_path = os.path.join(trainer.config['models_dir'], 'best_model_cnn_bilstm.pth')
    final_model_path = os.path.join(trainer.config['models_dir'], 'final_model_cnn_bilstm.pth')
    
    print(f"\nModel files saved at:")
    print(f"- Best model: {best_model_path}")
    print(f"- Final model: {final_model_path}")

if __name__ == "__main__":
    main()
