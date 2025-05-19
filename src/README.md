# NLP-Learning Project Structure

This directory contains the source code for the NLP-Learning project, organized in a modular structure for better maintainability and separation of concerns.

## Directory Structure

The project is organized into the following subdirectories:

### Core (`/src/core`)
Contains base classes and utilities used throughout the project:
- `nlpmodel.py` - Base NLP model class
- `common_imports.py` - Common imports used across modules
- `logging_config.py` - Logging configuration

### Data Processing (`/src/data`)
Contains data loading and processing modules:
- `data_processing.py` - General data processing functionality
- `preprocessing_model.py` - Text preprocessing logic

### Feature Engineering (`/src/features`)
Contains feature extraction and embedding modules:
- `embeddings.py` - Text embedding functionality
- `feature_engineering.py` - Legacy wrapper around modular components (for backward compatibility)

### Models (`/src/models`)
Contains model definition, training, and evaluation modules:
- `models.py` - Model architecture definitions
- `model_builder.py` - Model construction helpers
- `model_training.py` - Training loops and optimization
- `model_evaluation.py` - Evaluation metrics and visualization
- `test_models.py` - Test cases for models

### Visualization (`/src/visualization`)
Contains visualization and dashboard modules:
- `dashboard_generator.py` - Dashboard creation for model results

### Entry Points
The main entry points remain in the root directory:
- `main.py` - Main execution script
- `edamodel.py` - Exploratory Data Analysis functionality

## Usage

The modular structure allows you to import only the components you need:

```python
# Using specific components
from src.data.data_processing import DataProcessor
from src.features.embeddings import EmbeddingProcessor
from src.models.model_builder import ModelBuilder
from src.models.model_training import ModelTrainer

# Or for convenience, these are exposed at the top level too
from src import DataProcessor, EmbeddingProcessor, ModelBuilder, ModelTrainer
```

For backward compatibility, you can still use the feature_engineering wrapper:

```python
from src.features.feature_engineering import FeatureEngineering
```
