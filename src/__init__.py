# NLP-Learning package
# This file enables Python to recognize the directory as a package

# Import core modules
try:
    from src.core import NlpModel
    from src.core.common_imports import *  # noqa
    from src.core.logging_config import *  # noqa
except ImportError as e:
    print(f"Warning: Some core imports failed: {e}")

# Import data modules
try:
    from src.data import DataProcessor, PreprocessingModel
except ImportError as e:
    print(f"Warning: Some data imports failed: {e}")

# Import feature modules
try:
    from src.features import EmbeddingProcessor, FeatureEngineering
except ImportError as e:
    print(f"Warning: Some feature imports failed: {e}")

# Import model modules
try:
    from src.models import (
        SimpleLSTMModel, DeepLSTMModel, StackedLSTMModel, EnsembleModel,
        ModelBuilder, ModelTrainer, ModelEvaluator
    )
except ImportError as e:
    print(f"Warning: Some model imports failed: {e}")

# Import visualization modules
try:
    from src.visualization import DashboardGenerator
except ImportError as e:
    print(f"Warning: Some visualization imports failed: {e}")

__all__ = [
    # Core modules
    'NlpModel', 'common_imports', 'logging_config',
    
    # Data modules
    'DataProcessor', 'PreprocessingModel',
    
    # Feature modules
    'EmbeddingProcessor', 'FeatureEngineering',
    
    # Model modules
    'SimpleLSTMModel', 'DeepLSTMModel', 'StackedLSTMModel', 'EnsembleModel',
    'ModelBuilder', 'ModelTrainer', 'ModelEvaluator',
    
    # Visualization modules
    'DashboardGenerator',
    
    # Package modules
    'core', 'data', 'features', 'models', 'visualization'
]