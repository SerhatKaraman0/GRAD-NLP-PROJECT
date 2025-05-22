#!/bin/bash
# filepath: /Users/user/Desktop/Projects/NLP-Learning/run_advanced_kfold_training.sh

# Set up environment
echo "Setting up environment for advanced CNN-BiLSTM training with k-fold cross-validation..."

# Ensure we're in the project root
cd "$(dirname "$0")"

# Create directories if they don't exist
mkdir -p data/models

# Make scripts executable
chmod +x train_advanced_cnn_bilstm.py

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Display options
echo "=== Advanced CNN-BiLSTM Training Options ==="
echo "1. Single train/validation split"
echo "2. K-fold cross-validation (3 folds)"
echo "3. K-fold cross-validation (5 folds)"
echo "4. K-fold cross-validation (10 folds)"

read -p "Select an option (1-4): " option

case $option in
  1)
    echo "Running single train/validation split training..."
    python train_advanced_cnn_bilstm.py --no-kfold
    ;;
  2)
    echo "Running 3-fold cross-validation training..."
    python train_advanced_cnn_bilstm.py --kfold --folds 3
    ;;
  3)
    echo "Running 5-fold cross-validation training..."
    python train_advanced_cnn_bilstm.py --kfold --folds 5
    ;;
  4)
    echo "Running 10-fold cross-validation training..."
    python train_advanced_cnn_bilstm.py --kfold --folds 10
    ;;
  *)
    echo "Invalid option. Using default (5-fold cross-validation)."
    python train_advanced_cnn_bilstm.py --kfold
    ;;
esac

echo "Training process completed!"
echo "Check the logs for training results."
