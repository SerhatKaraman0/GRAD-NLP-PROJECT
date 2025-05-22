from src.core.common_imports import * # noqa: F403, F405
from src.data.data_processing import DataProcessor
from src.core.logging_config import *  # noqa: F403, F405

import numpy as np
import pandas as pd
import os
import gc
import io
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split

class EmbeddingProcessor(DataProcessor):
    """
    Class for handling text embeddings and feature preparation for NLP models
    """
    def __init__(self, batch_size=10000, max_features=10000, embedding_dim=100):
        """
        Initialize the EmbeddingProcessor
        
        Args:
            batch_size (int): Size of batch for processing
            max_features (int): Maximum number of features for vectorization  
            embedding_dim (int): Dimension of the word embeddings
        """
        super().__init__(batch_size=batch_size, max_features=max_features)
        self.embedding_dim = embedding_dim
        self.EMBEDDINGS_DIR = os.path.join(self.SAVE_DATA_DIR, "embeddings")
        os.makedirs(self.EMBEDDINGS_DIR, exist_ok=True)
    
    def prepare_embeddings(self, texts):
        """
        Prepare text embeddings for model input
        
        Args:
            texts (list): List of text strings to process
            
        Returns:
            tuple: (sequences, embedding_matrix)
        """
        self.logger.info("Preparing text embeddings...")
        
        # Ensure all texts are strings
        texts = [str(text) for text in texts]
        
        # Create a tokenizer with limited vocabulary size
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.max_features)
        tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        sequences = tokenizer.texts_to_sequences(texts)
        
        # Get word index mapping
        word_index = tokenizer.word_index
        self.logger.info(f"Found {len(word_index)} unique tokens")
        
        # Load pre-trained word embeddings (e.g., GloVe)
        embedding_matrix = self.load_embedding_matrix(word_index)
        
        return sequences, embedding_matrix
    
    def load_embedding_matrix(self, word_index):
        """
        Load pre-trained word embeddings and create an embedding matrix
        
        Args:
            word_index (dict): Dictionary mapping words to indices
            
        Returns:
            numpy.ndarray: Embedding matrix
        """
        # Path to GloVe embeddings
        glove_path = os.path.join(self.EMBEDDINGS_DIR, f'glove.6B.{self.embedding_dim}d.txt')
        
        # Check if embeddings exist
        if not os.path.exists(glove_path):
            self.logger.warning(f"Embeddings not found at {glove_path}")
            # Initialize a random embedding matrix
            vocab_size = min(len(word_index) + 1, self.max_features)
            embedding_matrix = np.random.normal(0, 0.1, (vocab_size, self.embedding_dim))
            return embedding_matrix
        
        self.logger.info(f"Loading word embeddings from {glove_path}...")
        
        # Load pre-trained embeddings
        embeddings_index = {}
        with open(glove_path, encoding='utf8') as f:
            for i, line in enumerate(tqdm(f, desc="Loading embeddings")):
                try:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
                except Exception as e:
                    self.logger.error(f"Error processing embedding line: {e}")
        
        self.logger.info(f"Found {len(embeddings_index)} word vectors.")
        
        # Create embedding matrix
        vocab_size = min(len(word_index) + 1, self.max_features)
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim))
        
        for word, i in tqdm(word_index.items(), desc="Creating embedding matrix"):
            if i >= self.max_features:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        
        # Save embedding matrix
        embedding_matrix_path = os.path.join(self.EMBEDDINGS_DIR, 'embedding_matrix.npy')
        np.save(embedding_matrix_path, embedding_matrix)
        self.logger.info(f"Embedding matrix saved to {embedding_matrix_path}")
        
        return embedding_matrix
    
    def prepare_data_for_training(self, texts, labels):
        """
        Prepare data for training by creating embeddings and splitting into train/test sets
        
        Args:
            texts (list): List of text strings
            labels (list): Corresponding labels
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test, embedding_matrix, max_len)
        """
        # Prepare embeddings
        sequences, embedding_matrix = self.prepare_embeddings(texts)
        
        # Pad sequences
        max_len = min(max(len(seq) for seq in sequences), 500)  # Cap at 500 tokens
        X = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        self.logger.info(f"Training data shape: {X_train.shape}")
        self.logger.info(f"Testing data shape: {X_test.shape}")
        self.logger.info(f"Score distribution: {np.bincount(labels.astype(int))}")
        
        return X_train, y_train, X_test, y_test, embedding_matrix, max_len
