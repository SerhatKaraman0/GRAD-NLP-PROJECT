# CNN BiLSTM Model for Sentiment Analysis

This directory contains scripts for training and evaluating a CNN BiLSTM hybrid model for sentiment analysis on review data.

## Model Architecture

The `CNNBiLSTMClassifier` combines convolutional neural networks (CNN) with bidirectional LSTM for effective sentiment classification. The architecture includes:

- **Embedding Layer**: Converts input text tokens to dense vectors using pre-trained GloVe embeddings
- **Spatial Dropout**: Reduces overfitting by dropping entire feature maps
- **1D Convolutional Layer**: Captures local patterns in the text
- **Max Pooling Layer**: Reduces dimensionality and extracts key features
- **Bidirectional LSTM**: Captures sequential context in both directions
- **Fully Connected Layers**: Maps features to sentiment classes

## Scripts

### Training

To train the model, run:

```bash
python run_cnn_bilstm_training.py [options]
```

#### Options:
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 64)
- `--lr`: Learning rate (default: 0.001)
- `--embedding_dim`: Embedding dimension (50, 100, 200 for GloVe) (default: 100)
- `--freeze_embeddings`: Freeze the pre-trained embedding layer during training
- `--patience`: Early stopping patience (default: 3)
- `--config`: Path to JSON configuration file

### Evaluation

To evaluate the trained model, run:

```bash
python src/models/test_cnn_lstm.py [options]
```

#### Options:
- `--model_path`: Path to the trained model
- `--data_path`: Path to the test data
- `--batch_size`: Batch size for evaluation
- `--embedding_dim`: Embedding dimension used in the model
- `--test_size`: Proportion of data to use for testing
- `--seed`: Random seed for reproducibility
- `--cpu`: Force using CPU even if GPU is available
- `--no_plot`: Disable result plotting
- `--no_save`: Disable saving results to files

## Model Outputs

Training produces the following files in the `data/models/` directory:
- `best_model_cnn_bilstm.pth`: Model with the best validation performance
- `final_model_cnn_bilstm.pth`: Model at the end of training
- `training_history_cnn_bilstm.json`: Training metrics history

Evaluation produces the following files in the `data/metrics/` directory:
- `cnn_bilstm_evaluation.png`: Visualizations of model performance
- `cnn_bilstm_metrics.csv`: Summary evaluation metrics
- `cnn_bilstm_predictions.csv`: Detailed predictions on test data

## Requirements

The model requires the following dependencies:
- PyTorch (>= 1.7.0)
- NumPy
- Pandas
- Scikit-learn
- Matplotlib (for visualizations)

## Data Format

The model expects input data in CSV format with at least two columns:
- `Text`: The review text
- `Score`: The rating (1-5)

The data should be preprocessed and normalized before training.

## Pre-trained Embeddings

This model uses pre-trained GloVe embeddings for better generalization and semantic understanding. GloVe (Global Vectors for Word Representation) embeddings are trained on a corpus of billions of words and capture semantic relationships between words.

### Available Embedding Dimensions:

- 50-dimensional embeddings (`glove.6B.50d.txt`)
- 100-dimensional embeddings (`glove.6B.100d.txt`) (default)
- 200-dimensional embeddings (`glove.6B.200d.txt`)

The pre-trained embeddings are loaded from the `data/embeddings/` directory. You can choose the embedding dimension using the `--embedding_dim` parameter when training the model.

### Fine-tuning Embeddings:

By default, the embedding layer is fine-tuned during training. To keep the embeddings fixed (frozen), use the `--freeze_embeddings` flag.

```bash
# Train with frozen 200-dimensional embeddings
python run_cnn_bilstm_training.py --embedding_dim 200 --freeze_embeddings
```
