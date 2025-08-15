# NLP-Learning: Advanced Sentiment Analysis with Deep Learning & GPU Acceleration

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)](https://tensorflow.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)

A comprehensive, production-ready NLP sentiment analysis system featuring state-of-the-art deep learning models, GPU acceleration, interactive web interfaces, and extensive visualization capabilities. This project demonstrates enterprise-level ML engineering practices with modular architecture, comprehensive testing, and deployment-ready containerization.

## 🚀 **Project Overview**

This project implements a complete end-to-end sentiment analysis pipeline with the following core capabilities:

### **🧠 Advanced Deep Learning Models**
- **CNN-BiLSTM Hybrid Architecture**: Combines convolutional and recurrent networks with attention mechanisms
- **Ensemble Methods**: Multiple model aggregation strategies for improved accuracy and robustness
- **Advanced Neural Architectures**: Gated CNNs, self-attention mechanisms, and residual connections
- **Model Variants**: Simple LSTM, Deep BiLSTM, Stacked LSTM, and sophisticated Ensemble models
- **Attention Mechanisms**: Self-attention and multi-head attention for enhanced text understanding

### **⚡ GPU Acceleration & High Performance Computing**
- **Custom CUDA Kernels**: Hand-optimized CUDA operations for text processing
- **Intelligent Memory Management**: Dynamic GPU memory allocation and monitoring
- **Optimized Batch Processing**: Adaptive batch sizing based on available GPU memory
- **Performance Benchmarking**: Comprehensive GPU vs CPU performance analysis tools
- **Memory Optimization**: Advanced memory management with garbage collection and caching

### **🔧 Advanced Natural Language Processing**
- **Multi-Library Integration**: spaCy, NLTK, TextBlob, and VADER sentiment analysis
- **Sophisticated Preprocessing**: Lemmatization, named entity recognition, spelling correction
- **Domain-Specific Processing**: Customizable text preprocessing for specific domains
- **Quality Control**: Text quality scoring, duplicate detection, and data validation
- **Multiple Embedding Support**: GloVe, FastText, Word2Vec, and custom domain-specific embeddings

### **📊 Comprehensive Visualization & Analysis**
- **Interactive HTML Dashboards**: Real-time model performance visualization
- **Web-Based Testing Interface**: Flask application for interactive model testing
- **Advanced Metrics Visualization**: Confusion matrices, error distributions, training curves
- **Comparative Analysis**: Side-by-side model performance comparison

### **🐳 Production-Ready Deployment**
- **Docker Containerization**: Optimized multi-stage Docker builds
- **Cross-Platform Support**: Windows (build.bat) and Unix (Makefile) build systems
- **Configuration Management**: JSON-based configuration system
- **Monitoring & Logging**: Comprehensive logging with configurable levels

## 📁 **Detailed Project Architecture**

```
NLP-Learning/                           # Root project directory
├── 📦 src/                             # Main source code package
│   ├── 🔧 core/                        # Core functionality and base classes
│   │   ├── nlpmodel.py                 # Base NLP model class
│   │   ├── common_imports.py           # Centralized imports
│   │   └── logging_config.py           # Advanced logging configuration
│   │
│   ├── 🗄️ data/                        # Data processing and management
│   │   ├── data_processing.py          # Core data processing functionality
│   │   ├── preprocessing_model.py      # CPU-based text preprocessing
│   │   ├── preprocessing_model_gpu.py  # GPU-accelerated preprocessing
│   │   ├── cuda_text_kernels.py        # Custom CUDA operations
│   │   ├── gpu_memory_manager.py       # GPU memory optimization
│   │   └── gpu_parallel_processor.py   # Parallel GPU processing
│   │
│   ├── 🔬 features/                    # Feature engineering and embeddings
│   │   ├── embeddings.py               # Base embedding functionality
│   │   ├── advanced_embeddings.py      # Advanced embedding techniques
│   │   ├── enhanced_text_preprocessing.py # Enhanced preprocessing
│   │   ├── feature_engineering.py      # Feature extraction
│   │   └── download_embeddings.py      # Embedding download utilities
│   │
│   ├── 🧠 models/                      # Model architectures and training
│   │   ├── models.py                   # Core model architectures
│   │   ├── advanced_models.py          # State-of-the-art architectures
│   │   ├── model_builder.py            # Model factory and construction
│   │   ├── model_training.py           # Advanced training procedures
│   │   ├── model_evaluation.py         # Comprehensive evaluation
│   │   └── test_models.py              # Model testing utilities
│   │
│   ├── 📈 visualization/               # Visualization and dashboards
│   │   ├── dashboard_generator.py      # HTML dashboard generation
│   │   └── model_comparison.py         # Model comparison utilities
│   │
│   └── main.py                         # Main execution pipeline
│
├── 🧪 tests/                           # Comprehensive testing suite
│   ├── model_test_interface.html       # Interactive web testing
│   ├── model_test_server.py            # Flask server for testing
│   ├── run_model_test_interface.py     # Test interface launcher
│   └── test_gpu_model.py              # GPU testing and benchmarking
│
├── 🗄️ data/                            # Data storage and artifacts
│   ├── embeddings/                     # Pre-trained embeddings
│   ├── metrics/                        # Performance metrics and plots
│   ├── models/                         # Saved model checkpoints
│   └── Reviews.csv                     # Primary dataset
│
├── 🚀 Training Scripts                 # Multiple training options
│   ├── run_cnn_bilstm_training.py     # Basic CNN-BiLSTM training
│   ├── improved_cnn_bilstm.py         # Improved architectures
│   ├── train_advanced_cnn_bilstm.py   # Advanced with k-fold CV
│   └── run_*.sh                       # Interactive training scripts
│
├── 🐳 Dockerfile                      # Container configuration
├── requirements.txt                    # Python dependencies
├── Makefile & build.bat               # Build systems
└── README.md                          # This documentation
```

## 🛠️ **Installation Guide**

### **System Requirements**
- **Python**: 3.8+ (recommended: 3.11)
- **RAM**: 8GB+ (16GB+ recommended)
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional but recommended)
- **CUDA**: 11.8+ for GPU acceleration

### **Quick Setup**

#### **Unix/Linux/macOS:**
```bash
git clone <repository-url>
cd NLP-Learning
make setup
source venv/bin/activate
python -m spacy download en_core_web_sm
```

#### **Windows:**
```cmd
git clone <repository-url>
cd NLP-Learning
build.bat setup
.\venv\Scripts\activate
python -m spacy download en_core_web_sm
```

#### **Docker:**
```bash
docker build -t nlp-sentiment .
docker run -d --name nlp-container --gpus all -p 8080:8080 nlp-sentiment
```

## 🚀 **Quick Start**

### **1. Complete Pipeline**
```bash
# Run full sentiment analysis pipeline
python -m src.main
```

### **2. Interactive Web Interface**
```bash
# Launch web testing interface
python tests/run_model_test_interface.py
# Open browser to http://localhost:5000
```

### **3. Custom Training**
```bash
# Basic CNN-BiLSTM training
python run_cnn_bilstm_training.py

# Advanced training with k-fold CV
./run_advanced_kfold_training.sh

# GPU performance benchmarking
python benchmark_gpu_performance.py
```

## 📚 **API Documentation**

### **Core Classes**

#### **DataProcessor**
```python
class DataProcessor(NlpModel):
    def __init__(self, batch_size=10000, max_features=10000):
        """Initialize with memory optimization settings"""
    
    def load_and_process_data(self):
        """Load and prepare data for training"""
        # Returns: texts, labels
    
    def create_bow(self):
        """Create optimized bag-of-words representation"""
        # Memory-efficient sparse matrix processing
```

#### **EmbeddingProcessor**
```python
class EmbeddingProcessor(DataProcessor):
    def __init__(self, batch_size=10000, max_features=10000, embedding_dim=100):
        """Initialize embedding processor with GloVe support"""
    
    def prepare_data_for_training(self, texts, labels):
        """Complete data preparation pipeline"""
        # Returns: X_train, y_train, X_test, y_test, embedding_matrix, max_len
    
    def load_embedding_matrix(self, word_index):
        """Load pre-trained GloVe embeddings"""
        # Supports 50d, 100d, 200d, 300d dimensions
```

#### **Model Architectures**

**SimpleLSTMModel**
```python
class SimpleLSTMModel(nn.Module):
    """Bidirectional LSTM with embedding layer"""
    # Features: Embedding → BiLSTM → Dropout → Linear
```

**CNNBiLSTMClassifier**
```python
class CNNBiLSTMClassifier(nn.Module):
    """Advanced CNN-BiLSTM with attention"""
    # Features: Multi-kernel CNN → BiLSTM → Attention → Dense layers
```

**AdvancedCNNBiLSTMClassifier**
```python
class AdvancedCNNBiLSTMClassifier(nn.Module):
    """State-of-the-art architecture with gated CNNs"""
    # Features: Gated CNNs → Multi-layer BiLSTM → Self-attention
```

#### **Training & Evaluation**

**ModelTrainer**
```python
class ModelTrainer:
    def train_model(self, X, y, embedding_matrix, max_len, model_type='ensemble'):
        """Advanced training with early stopping, LR scheduling"""
        # Features: Progress tracking, GPU optimization, checkpointing
```

**ModelEvaluator**
```python
class ModelEvaluator:
    def evaluate_model(self, model, X_test, y_test, model_type='ensemble'):
        """Comprehensive evaluation with visualizations"""
        # Returns: MSE, MAE, accuracy, predictions, detailed_results
```

## 🎯 **Available Commands**

### **Training**
```bash
python -m src.main                      # Complete pipeline
python run_cnn_bilstm_training.py       # Basic CNN-BiLSTM
python improved_cnn_bilstm.py           # Improved architecture
./run_advanced_kfold_training.sh        # K-fold cross-validation
```

### **Testing**
```bash
python tests/run_model_test_interface.py # Web interface
python tests/test_gpu_model.py          # GPU testing
python benchmark_gpu_performance.py     # Performance analysis
```

### **Data Processing**
```bash
make run-preprocessing                   # Run preprocessing
make run-feature_eng                    # Feature engineering
python src/features/download_embeddings.py # Download embeddings
```

### **Development**
```bash
make setup / build.bat setup            # Environment setup
make clean / build.bat clean            # Clean environment
make test / build.bat test              # Run tests
make build / build.bat build            # Docker build
```

## ⚙️ **Configuration**

### **Environment Variables**
```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export LOG_LEVEL=INFO
```

### **Model Configuration**
```json
{
    "model_architecture": {
        "type": "advanced_cnn_bilstm",
        "embedding_dim": 200,
        "vocab_size": 25000,
        "dropout_rate": 0.5,
        "lstm_hidden_size": 256
    },
    "training_config": {
        "batch_size": 64,
        "epochs": 25,
        "learning_rate": 0.001,
        "early_stopping_patience": 7
    }
}
```

## 📈 **Performance Benchmarks**

| Model Type | Accuracy | F1-Score | Training Time | GPU Memory |
|------------|----------|----------|---------------|------------|
| Simple LSTM | 87.2% | 0.871 | 15 min | 2.1 GB |
| Deep BiLSTM | 89.5% | 0.894 | 25 min | 3.2 GB |
| CNN-BiLSTM | 91.3% | 0.912 | 35 min | 4.1 GB |
| Advanced CNN-BiLSTM | 92.7% | 0.926 | 45 min | 5.2 GB |
| Ensemble | 93.1% | 0.930 | 60 min | 6.8 GB |

### **GPU Acceleration Benefits**
- **Preprocessing**: 3-5x speedup vs CPU
- **Training**: 8-12x speedup vs CPU
- **Inference**: 15-20x speedup vs CPU

## 🔧 **Troubleshooting**

### **GPU Memory Issues**
```bash
# Reduce batch size
python run_cnn_bilstm_training.py --batch_size 32

# Set memory fraction
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### **Model Loading Issues**
```bash
# Check compatibility
python src/models/test_models.py --check_compatibility

# Rebuild models
python -m src.main --rebuild_models
```

## 📚 **Additional Documentation**

- [Model Testing Guide](tests/README_MODEL_TESTING.md)
- [CNN-BiLSTM Documentation](src/models/README_CNN_BILSTM.md)
- [Visualization Features](docs/visualization_features.md)

## 🚀 **Deployment**

### **Docker Production**
```bash
docker run -d --name nlp-prod \
  --gpus all \
  -p 80:8080 \
  -v /data:/app/data \
  -e ENVIRONMENT=production \
  nlp-sentiment:latest
```

### **Cloud Deployment**
Supports AWS, GCP, Azure with GPU instances. See Docker Compose configurations for scalable deployments.

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Install dev dependencies: `pip install -r requirements-dev.txt`
4. Make changes and add tests
5. Run tests: `make test`
6. Submit pull request

## 📄 **License**

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- PyTorch and TensorFlow teams
- Hugging Face for transformers
- spaCy for NLP capabilities
- Stanford NLP Group for GloVe
- NVIDIA for CUDA support

---

**⭐ Star this repository if you find it useful!**

**Made with ❤️ for the NLP and Machine Learning community**
