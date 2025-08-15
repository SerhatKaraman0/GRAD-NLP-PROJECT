#!/bin/bash
# filepath: /Users/user/Desktop/Projects/NLP-Learning/run_advanced_training.sh

# Set up environment
echo "Setting up environment..."

# Ensure we're in the project root
cd "$(dirname "$0")"

# Create directories if they don't exist
mkdir -p config data/models

# Make scripts executable
chmod +x train_advanced_cnn_bilstm.py
chmod +x src/features/enhanced_text_preprocessing.py

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Step 1: Enhance data preprocessing (if needed)
read -p "Do you want to run enhanced text preprocessing? (y/n): " run_preprocessing
if [[ $run_preprocessing == "y" ]]; then
  echo "Running enhanced text preprocessing..."
  python src/features/enhanced_text_preprocessing.py \
    --input data/Reviews.csv \
    --output data/ENHANCED_Reviews.csv \
    --text-column Text \
    --output-column ProcessedText
  
  # Update config to use enhanced data
  cat > config/advanced_cnn_bilstm.json << EOF
{
    "data_path": "data/ENHANCED_Reviews.csv",
    "text_column": "ProcessedText",
    "models_dir": "data/models",
    "embedding_dim": 200,
    "max_features": 25000,
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 0.0005,
    "validation_split": 0.15,
    "max_seq_length": 300,
    "num_classes": 5,
    "random_state": 42,
    "early_stopping_patience": 5,
    "dropout_rate": 0.5,
    "spatial_dropout": 0.4,
    "lstm_hidden_size": 256,
    "lstm_layers": 2,
    "weight_decay": 0.0001,
    "gradient_clip": 1.0,
    "label_smoothing": 0.2,
    "use_mixed_precision": true
}
EOF
else
  # Config already exists - we'll use it
  echo "Using existing config file: config/advanced_cnn_bilstm.json"
fi

# Step 2: Train the advanced model
echo "Starting advanced CNN-BiLSTM training..."
python train_advanced_cnn_bilstm.py

echo "Training process completed!"
echo "Check the logs for training results."
