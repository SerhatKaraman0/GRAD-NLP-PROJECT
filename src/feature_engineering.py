from src.common_imports import *  # noqa: F403, F405
from src.nlpmodel import NlpModel
from src.logging_config import *  # noqa: F403, F405
from utils.helper import CONTRACTIONS_DICT, SLANG_DICT  
from src.models import SimpleLSTMModel, DeepLSTMModel, StackedLSTMModel, EnsembleModel

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import gc
import sys
import logging
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    precision_score, recall_score, accuracy_score, 
    mean_squared_error, mean_absolute_error, r2_score
)
import io
import base64

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import wget
import zipfile
import json
from gensim.models import Word2Vec

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineering(NlpModel):
    def __init__(self, batch_size=10_000, max_features=7000, embedding_dim=100):
        """Initialize paths and load dataset."""
        super().__init__()
        self.SAVE_DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.PREPROCESSED_DATA_PATH = os.path.join(self.SAVE_DATA_DIR, "PREPROCESSED_Reviews.csv")
        self.EMBEDDING_PATH = os.path.join(self.SAVE_DATA_DIR, "embeddings")
        self.batch_size = batch_size
        self.max_features = max_features
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.embedding_matrix = None
        os.makedirs(self.EMBEDDING_PATH, exist_ok=True)

    def prepare_embeddings(self, texts):
        """
        Prepare word embeddings using Word2Vec and GloVe.
        
        Args:
            texts: List of preprocessed text documents
            
        Returns:
            embedding_matrix: Pre-trained word embeddings
        """
        logger.info("Preparing word embeddings...")
        
        # Clean and tokenize texts
        cleaned_texts = []
        for text in texts:
            # Convert to string and clean
            text = str(text).lower()
            # Split into words and remove empty strings
            words = [word.strip() for word in text.split() if word.strip()]
            cleaned_texts.append(' '.join(words))
        
        # Tokenize texts
        logger.info("Tokenizing texts...")
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=self.max_features,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' ',
            oov_token='<UNK>'
        )
        self.tokenizer.fit_on_texts(cleaned_texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        
        # Get vocabulary info
        word_index = self.tokenizer.word_index
        logger.info(f"Found {len(word_index)} unique tokens")
        
        # Load pre-trained GloVe embeddings
        logger.info("Loading GloVe embeddings...")
        embeddings_index = {}
        glove_path = os.path.join(self.EMBEDDING_PATH, 'glove.6B.100d.txt')
        
        if not os.path.exists(glove_path):
            logger.info("Downloading GloVe embeddings...")
            # Download GloVe embeddings
            url = 'http://nlp.stanford.edu/data/glove.6B.zip'
            wget.download(url, self.EMBEDDING_PATH)
            with zipfile.ZipFile(os.path.join(self.EMBEDDING_PATH, 'glove.6B.zip'), 'r') as zip_ref:
                zip_ref.extractall(self.EMBEDDING_PATH)
        
        with open(glove_path, encoding='utf-8') as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, 'f', sep=' ')
                embeddings_index[word] = coefs
        
        logger.info(f"Loaded {len(embeddings_index)} word vectors from GloVe")
        
        # Create embedding matrix
        logger.info("Creating embedding matrix...")
        self.embedding_matrix = np.zeros((self.max_features, self.embedding_dim))
        for word, i in word_index.items():
            if i >= self.max_features:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector
        
        # Train Word2Vec on our corpus
        logger.info("Training Word2Vec on corpus...")
        tokenized_texts = [text.split() for text in cleaned_texts]
        word2vec_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.embedding_dim,
            window=5,
            min_count=1,
            workers=4
        )
        
        # Combine GloVe and Word2Vec embeddings
        logger.info("Combining GloVe and Word2Vec embeddings...")
        words_found = 0
        for word, i in word_index.items():
            if i >= self.max_features:
                continue
            if self.embedding_matrix[i].sum() == 0 and word in word2vec_model.wv:
                self.embedding_matrix[i] = word2vec_model.wv[word]
                words_found += 1
        
        logger.info(f"Added {words_found} word vectors from Word2Vec")
        
        return sequences, self.embedding_matrix

    def load_and_process_data(self):
        """Load and process the data including embedding preparation."""
        logger.info("Loading and processing data...")
        
        # Load preprocessed data
        df = pd.read_csv(self.PREPROCESSED_DATA_PATH)
        
        # Convert text to string and handle NaN values
        df['Text'] = df['Text'].fillna('').astype(str)
        
        # Map scores to sentiment (1-2: negative, 3: neutral, 4-5: positive)
        df['sentiment'] = df['Score'].apply(lambda x: 0 if x <= 2 else (2 if x >= 4 else 1))
        
        # Remove any rows with empty text
        df = df[df['Text'].str.strip() != '']
        
        # Use the 'Text' column for our analysis
        texts = df['Text'].values
        labels = df['sentiment'].values
        
        logger.info(f"Total samples: {len(texts)}")
        logger.info(f"Sample text: {texts[0][:100]}...")
        
        # Prepare embeddings
        sequences, embedding_matrix = self.prepare_embeddings(texts)
        
        # Pad sequences
        max_len = min(max(len(seq) for seq in sequences), 500)  # Cap at 500 tokens
        X = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Testing data shape: {X_test.shape}")
        logger.info(f"Label distribution: {np.bincount(labels)}")
        
        return X_train, y_train, X_test, y_test, embedding_matrix, max_len

    def train_model(self, X, y, embedding_matrix, max_len, model_type='ensemble'):
        """
        Train the specified model type with embeddings.
        
        Args:
            X: Training features
            y: Target values
            embedding_matrix: Pre-trained word embeddings
            max_len: Maximum sequence length
            model_type: Type of model to train
            
        Returns:
            tf.keras.models.Model: Trained model
        """
        # Select model type
        if model_type == 'simple':
            model_class = SimpleLSTMModel(
                max_features=self.max_features,
                embedding_dim=self.embedding_dim,
                max_len=max_len,
                embedding_matrix=embedding_matrix
            )
        elif model_type == 'deep':
            model_class = DeepLSTMModel(
                max_features=self.max_features,
                embedding_dim=self.embedding_dim,
                max_len=max_len,
                embedding_matrix=embedding_matrix
            )
        elif model_type == 'stacked':
            model_class = StackedLSTMModel(
                max_features=self.max_features,
                embedding_dim=self.embedding_dim,
                max_len=max_len,
                embedding_matrix=embedding_matrix
            )
        else:
            model_class = EnsembleModel(
                max_features=self.max_features,
                embedding_dim=self.embedding_dim,
                max_len=max_len,
                embedding_matrix=embedding_matrix
            )
        
        # Build and train model
        model = model_class.build()
        
        # Setup callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.SAVE_DATA_DIR, f'best_model_{model_type}.h5'),
                monitor='val_loss',
                save_best_only=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.SAVE_DATA_DIR, 'logs', model_type),
                histogram_freq=1
            )
        ]
        
        # Train model
        logger.info(f"Training {model_type} model")
        history = model.fit(
            X, y,
            epochs=20,
            batch_size=256,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training history
        self.save_training_history(history, model_type)
        
        return model, history

    def save_training_history(self, history, model_type):
        """Save training history to file."""
        history_path = os.path.join(self.SAVE_DATA_DIR, f'training_history_{model_type}.json')
        with open(history_path, 'w') as f:
            json.dump(history.history, f)
        logger.info(f"Training history saved to {history_path}")

    def build_model(self, input_dim):
        """
        Select and build a model for sentiment analysis.
        
        Args:
            input_dim: Dimension of the input features
            
        Returns:
            Model: Selected model instance
        """
        # Create ensemble model by default
        model = EnsembleModel(input_dim)
        return model.build()

    def evaluate_model(self, model, X_test, y_test, model_type='ensemble'):
        """
        Evaluate the model on test data.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target values
            model_type: Type of model being evaluated
            
        Returns:
            tuple: (loss, mean absolute error, predicted values)
        """
        logger.info("Evaluating model on test data")
        
        # Prepare inputs based on model type
        if model_type == 'ensemble':
            X_test_input = [X_test, X_test, X_test]
        else:
            X_test_input = X_test
        
        # Evaluate model
        metrics = model.evaluate(X_test_input, y_test, verbose=0)
        loss, mae, acc = metrics
        logger.info(f"Model evaluation: Loss={loss:.4f}, MAE={mae:.4f}, Accuracy={acc:.4f}")
        
        # Get predictions
        y_pred = model.predict(X_test_input, verbose=0).flatten()
        
        # Create visualizations and save metrics
        self._create_evaluation_visualizations(y_test, y_pred, mae)
        
        return loss, mae, y_pred

    def _create_evaluation_visualizations(self, y_test, y_pred, mae):
        """Create and save evaluation visualizations."""
        metrics_dir = os.path.join(self.SAVE_DATA_DIR, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Create visualizations...
        # [Previous visualization code remains the same]

def main():
    """Main execution function."""
    # Enable memory growth for GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU(s) detected: {len(gpus)}")
        except Exception as e:
            logger.warning(f"Error setting GPU memory growth: {e}")
    
    # Initialize FeatureEngineering class
    model = FeatureEngineering(batch_size=10_000, max_features=7000, embedding_dim=100)
    
    try:
        # Load data and prepare embeddings
        X_train, y_train, X_test, y_test, embedding_matrix, max_len = model.load_and_process_data()
        
        # Train and evaluate each model type separately
        model_types = ['simple', 'deep', 'stacked', 'ensemble']
        results = {}
        
        for model_type in model_types:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_type.upper()} LSTM Model")
            logger.info(f"{'='*50}")
            
            # Train the model
            trained_model, history = model.train_model(
                X_train, y_train,
                embedding_matrix=embedding_matrix,
                max_len=max_len,
                model_type=model_type
            )
            
            # Evaluate the model
            metrics = trained_model.evaluate(X_test, y_test, verbose=0)
            y_pred = trained_model.predict(X_test)
            
            # Calculate additional metrics
            results[model_type] = {
                'loss': metrics[0],
                'accuracy': metrics[1],
                'precision': precision_score(y_test, y_pred.round()),
                'recall': recall_score(y_test, y_pred.round()),
                'f1': f1_score(y_test, y_pred.round())
            }
            
            # Save model and results
            model_path = os.path.join(model.SAVE_DATA_DIR, f"sentiment_model_{model_type}.h5")
            trained_model.save(model_path)
            
            # Clear memory
            del trained_model
            gc.collect()
            if gpus:
                tf.keras.backend.clear_session()
        
        # Compare results
        results_df = pd.DataFrame(results).round(4)
        results_path = os.path.join(model.SAVE_DATA_DIR, 'model_comparison.csv')
        results_df.to_csv(results_path)
        logger.info(f"\nModel Comparison Results:\n{results_df}")
        
        logger.info("\nAll models have been trained and evaluated successfully!")
        
    except Exception as e:
        logger.error(f"Error in processing: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
