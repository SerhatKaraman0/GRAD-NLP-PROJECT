# NLP-Learning Project Refactoring

This project has been refactored to improve code organization, maintainability, and reusability.

## Project Structure

The codebase has been divided into multiple modules:

- `data_processing.py`: Data loading, batch processing, and data manipulation
- `embeddings.py`: Text embedding and feature preparation
- `model_builder.py`: Model architecture definitions and factory methods
- `model_training.py`: Training loop, optimization, and history tracking
- `model_evaluation.py`: Metrics computation and evaluation visualization
- `dashboard_generator.py`: HTML dashboard generation for results visualization
- `models.py`: Model class definitions
- `main.py`: Entry point for running the full pipeline

## How to Run

To run the full sentiment analysis pipeline:

```bash
python -m src.main
```

## Individual Components

You can also use individual components for specific tasks:

```python
from src.data_processing import DataProcessor
from src.embeddings import EmbeddingProcessor
from src.model_builder import ModelBuilder
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.dashboard_generator import DashboardGenerator

# Example: Process data and create embeddings
processor = EmbeddingProcessor(batch_size=10000, max_features=7000, embedding_dim=100)
texts, labels = processor.load_and_process_data()
X_train, y_train, X_test, y_test, embedding_matrix, max_len = processor.prepare_data_for_training(texts, labels)

# Example: Train a model
trainer = ModelTrainer('/path/to/save/models')
model, history = trainer.train_model(X_train, y_train, embedding_matrix, max_len, model_type='simple')

# Example: Evaluate a model
evaluator = ModelEvaluator('/path/to/save/results')
mse, mae, acc, y_pred, results = evaluator.evaluate_model(model, X_test, y_test, model_type='simple')
```

## Dependencies

This project requires the following libraries:
- PyTorch
- TensorFlow
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- tqdm
- and others listed in requirements.txt
