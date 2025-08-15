#!/bin/bash
# filepath: /Users/user/Desktop/Projects/NLP-Learning/run_improved_training.sh

# Set up environment
echo "Setting up environment for improved CNN-BiLSTM training..."

# Ensure we're in the project root
cd "$(dirname "$0")"

# Create directories if they don't exist
mkdir -p data/models

# Make scripts executable
chmod +x improved_cnn_bilstm.py

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Display options
echo "=== Improved CNN-BiLSTM Training Options ==="
echo "1. Single train/validation split"
echo "2. K-fold cross-validation (3 folds)"
echo "3. K-fold cross-validation (5 folds)"

read -p "Select an option (1-3): " option

case $option in
  1)
    echo "Running single train/validation split training..."
    python improved_cnn_bilstm.py --epochs 15 --batch-size 32 --embedding-dim 100
    ;;
  2)
    echo "Running 3-fold cross-validation training..."
    python improved_cnn_bilstm.py --kfold --folds 3 --epochs 15 --batch-size 32 --embedding-dim 100
    ;;
  3)
    echo "Running 5-fold cross-validation training..."
    python improved_cnn_bilstm.py --kfold --folds 5 --epochs 15 --batch-size 32 --embedding-dim 100
    ;;
  *)
    echo "Invalid option. Using default (single split)."
    python improved_cnn_bilstm.py
    ;;
esac

echo "Training process completed!"
echo "Check the logs for training results."
