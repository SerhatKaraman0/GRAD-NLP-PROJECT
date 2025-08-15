#!/bin/zsh
# filepath: /Users/user/Desktop/Projects/NLP-Learning/experiments/run_embedding_experiments.sh

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Change to project root directory
cd "$(dirname "$0")/.."

# Ensure directory exists
mkdir -p experiments/results

echo -e "${BLUE}Starting CNN-BiLSTM experiments with different GloVe embeddings${NC}"

# Make sure we have the embeddings
echo -e "${BLUE}Checking for GloVe embeddings...${NC}"
python src/features/download_embeddings.py

# Define experiment configurations
EXPERIMENT_NAME="glove_embedding_comparison"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="experiments/results/${EXPERIMENT_NAME}_${TIMESTAMP}.log"

# Run experiments with different embedding dimensions
echo -e "${GREEN}Running embeddings experiment - logging to $LOG_FILE${NC}"

# Function to run experiment with specific parameters
run_experiment() {
    dim=$1
    freeze=$2
    epochs=$3
    
    freeze_flag=""
    freeze_name="fine-tuned"
    if [[ "$freeze" == "true" ]]; then
        freeze_flag="--freeze_embeddings"
        freeze_name="frozen"
    fi
    
    echo -e "\n${BLUE}=====================================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${GREEN}Running experiment with ${dim}d ${freeze_name} embeddings${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}=====================================================${NC}\n" | tee -a "$LOG_FILE"
    
    # Run training
    python run_cnn_bilstm_training.py --embedding_dim $dim $freeze_flag --epochs $epochs --batch_size 64 | tee -a "$LOG_FILE"
    
    # Check for the trained model
    model_path="data/models/best_model_cnn_bilstm.pth"
    if [[ -f "$model_path" ]]; then
        # Create a copy with experiment-specific name
        model_copy="data/models/cnn_bilstm_${dim}d_${freeze_name}.pth"
        cp "$model_path" "$model_copy"
        echo -e "\n${GREEN}Model saved as ${model_copy}${NC}" | tee -a "$LOG_FILE"
        
        # Run evaluation
        echo -e "\n${BLUE}Evaluating model...${NC}\n" | tee -a "$LOG_FILE"
        python src/models/test_cnn_lstm.py --model_path "$model_copy" --embedding_dim $dim | tee -a "$LOG_FILE"
    else
        echo -e "\n${RED}ERROR: Model training failed - model file not found${NC}" | tee -a "$LOG_FILE"
    fi
}

# Run experiments with different dimensions
# Dimension, Freeze (true/false), Epochs
run_experiment 50 false 5
run_experiment 100 false 5
run_experiment 200 false 5

# Run experiments with frozen embeddings
run_experiment 100 true 5

echo -e "\n${GREEN}All experiments completed. Results saved to $LOG_FILE${NC}"
