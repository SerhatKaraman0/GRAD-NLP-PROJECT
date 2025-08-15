#!/usr/bin/env python
# filepath: /Users/user/Desktop/Projects/NLP-Learning/src/features/advanced_embeddings.py

"""
Enhanced embeddings processor with advanced pretraining capabilities.
Supports multiple embedding types, contextual embeddings, and domain-specific fine-tuning.
"""

from src.core.common_imports import * # noqa: F403, F405
from src.features.embeddings import EmbeddingProcessor
from src.core.logging_config import *  # noqa: F403, F405

import numpy as np
import pandas as pd
import os
import gc
import io
import pickle
import time
from tqdm import tqdm
from collections import Counter
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

class AdvancedEmbeddingProcessor(EmbeddingProcessor):
    """
    Enhanced embedding processor with advanced pretraining capabilities.
    Supports multiple embedding types, contextual embeddings, and fine-tuning.
    """
    
    EMBEDDING_TYPES = {
        'glove': 'GloVe embeddings',
        'fasttext': 'FastText embeddings',
        'word2vec': 'Word2Vec embeddings',
        'domain-specific': 'Domain-specific embeddings'
    }
    
    def __init__(self, batch_size=10000, max_features=20000, embedding_dim=100, 
                 embedding_type='glove', use_subword=False, cache_dir=None):
        """
        Initialize the AdvancedEmbeddingProcessor
        
        Args:
            batch_size (int): Size of batch for processing
            max_features (int): Maximum number of features for tokenization
            embedding_dim (int): Dimension of the word embeddings
            embedding_type (str): Type of embeddings to use (glove, fasttext, word2vec, domain-specific)
            use_subword (bool): Whether to use subword tokenization
            cache_dir (str): Directory to cache processed embeddings
        """
        super().__init__(batch_size=batch_size, max_features=max_features, embedding_dim=embedding_dim)
        
        self.embedding_type = embedding_type if embedding_type in self.EMBEDDING_TYPES else 'glove'
        self.use_subword = use_subword
        self.tokenizer = None
        
        # Set up caching for faster iterations
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = os.path.join(self.EMBEDDINGS_DIR, 'cache')
        
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger.info(f"Using {self.EMBEDDING_TYPES[self.embedding_type]} with dimension {embedding_dim}")
    
    def create_tokenizer(self, texts, use_cache=True):
        """
        Create and fit a tokenizer on the provided texts with caching
        
        Args:
            texts (list): List of text documents to tokenize
            use_cache (bool): Whether to use cached tokenizer if available
            
        Returns:
            tensorflow.keras.preprocessing.text.Tokenizer: Fitted tokenizer
        """
        cache_path = os.path.join(self.cache_dir, f'tokenizer_{self.max_features}.pkl')
        
        # Try to load from cache if requested
        if use_cache and os.path.exists(cache_path):
            self.logger.info("Loading tokenizer from cache...")
            with open(cache_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            return self.tokenizer
            
        # Create a new tokenizer
        self.logger.info("Creating new tokenizer...")
        self.tokenizer = Tokenizer(num_words=self.max_features, 
                                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True, split=' ', oov_token='<OOV>')
        
        start_time = time.time()
        self.tokenizer.fit_on_texts(texts)
        
        # Save to cache
        with open(cache_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
            
        self.logger.info(f"Tokenizer created in {time.time() - start_time:.2f} seconds")
        self.logger.info(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        return self.tokenizer
    
    def prepare_advanced_embeddings(self, texts, use_cache=True):
        """
        Prepare advanced text embeddings with better preprocessing and handling
        
        Args:
            texts (list): List of text strings to process
            use_cache (bool): Whether to use cached results if available
            
        Returns:
            tuple: (sequences, embedding_matrix, word_index)
        """
        cache_path = os.path.join(self.cache_dir, 
                                 f'embeddings_{self.embedding_type}_{self.embedding_dim}_{self.max_features}.npz')
        
        if use_cache and os.path.exists(cache_path):
            self.logger.info("Loading embeddings from cache...")
            cache_data = np.load(cache_path, allow_pickle=True)
            sequences = cache_data['sequences']
            embedding_matrix = cache_data['embedding_matrix']
            word_index = cache_data['word_index'].item()  # Convert from 0d array to dict
            self.logger.info(f"Loaded embedding matrix shape: {embedding_matrix.shape}")
            return sequences, embedding_matrix, word_index
        
        self.logger.info("Preparing advanced text embeddings...")
        
        # Create or load tokenizer
        if self.tokenizer is None:
            self.create_tokenizer(texts, use_cache)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        word_index = self.tokenizer.word_index
        self.logger.info(f"Found {len(word_index)} unique tokens")
        
        # Load appropriate embedding matrix
        if self.embedding_type == 'domain-specific':
            embedding_matrix = self.create_domain_specific_embeddings(texts, word_index)
        else:
            embedding_matrix = self.load_embedding_matrix(word_index)
        
        # Save to cache
        np.savez(cache_path, 
                 sequences=sequences, 
                 embedding_matrix=embedding_matrix, 
                 word_index=word_index)
        
        return sequences, embedding_matrix, word_index
    
    def load_embedding_matrix(self, word_index):
        """
        Enhanced method to load pre-trained word embeddings with better handling
        
        Args:
            word_index (dict): Dictionary mapping words to indices
            
        Returns:
            numpy.ndarray: Embedding matrix
        """
        # Determine embedding file path based on type
        if self.embedding_type == 'glove':
            embedding_path = os.path.join(self.EMBEDDINGS_DIR, f'glove.6B.{self.embedding_dim}d.txt')
            binary = False
        elif self.embedding_type == 'fasttext':
            embedding_path = os.path.join(self.EMBEDDINGS_DIR, f'crawl-300d-2M.vec')
            binary = False
        elif self.embedding_type == 'word2vec':
            embedding_path = os.path.join(self.EMBEDDINGS_DIR, 'GoogleNews-vectors-negative300.bin')
            binary = True
        else:
            self.logger.warning(f"Unknown embedding type: {self.embedding_type}, falling back to GloVe")
            embedding_path = os.path.join(self.EMBEDDINGS_DIR, f'glove.6B.{self.embedding_dim}d.txt')
            binary = False
        
        # Check if embeddings exist
        if not os.path.exists(embedding_path):
            self.logger.warning(f"Embeddings not found at {embedding_path}")
            self.logger.info("Initializing random embedding matrix")
            return self._initialize_random_embeddings(word_index)
        
        self.logger.info(f"Loading embeddings from {embedding_path}...")
        start_time = time.time()
        
        # Load embeddings
        embeddings_index = {}
        if not binary:
            with open(embedding_path, encoding='utf8') as f:
                for i, line in enumerate(tqdm(f, desc=f"Loading {self.embedding_type} embeddings")):
                    try:
                        values = line.split()
                        word = values[0]
                        coefs = np.asarray(values[1:], dtype='float32')
                        embeddings_index[word] = coefs
                    except Exception as e:
                        self.logger.error(f"Error processing embedding line: {e}")
        else:
            # Binary format handling for Word2Vec
            try:
                import gensim
                model = gensim.models.KeyedVectors.load_word2vec_format(
                    embedding_path, binary=True, limit=1000000)
                for word in tqdm(model.index_to_key, desc="Converting Word2Vec model"):
                    embeddings_index[word] = model[word]
            except ImportError:
                self.logger.error("Gensim not installed, needed for binary embedding formats")
                return self._initialize_random_embeddings(word_index)
        
        self.logger.info(f"Found {len(embeddings_index)} word vectors in {time.time() - start_time:.2f} seconds")
        
        # Create embedding matrix with special token handling
        vocab_size = min(len(word_index) + 1, self.max_features)
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim))
        
        # Add special tokens
        special_tokens = {
            '<PAD>': np.zeros(self.embedding_dim),
            '<OOV>': np.random.normal(0, 0.1, self.embedding_dim),
            '<START>': np.random.normal(0, 0.1, self.embedding_dim),
            '<END>': np.random.normal(0, 0.1, self.embedding_dim)
        }
        
        # Add embeddings for each word
        num_words_found = 0
        for word, i in tqdm(word_index.items(), desc=f"Building {self.embedding_type} embedding matrix"):
            if i >= vocab_size:
                continue
                
            # Try to get embedding for the word
            embedding_vector = None
            
            # Check if it's a special token
            if word in special_tokens:
                embedding_vector = special_tokens[word]
            else:
                # Try the word as is
                embedding_vector = embeddings_index.get(word)
                
                # If not found, try lowercase
                if embedding_vector is None:
                    embedding_vector = embeddings_index.get(word.lower())
                
                # Try word without punctuation
                if embedding_vector is None and not self.use_subword:
                    word_clean = ''.join(c for c in word if c.isalnum())
                    if word_clean:
                        embedding_vector = embeddings_index.get(word_clean)
            
            # Set the embedding
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                num_words_found += 1
            else:
                # Initialize OOV words with random values from normal distribution
                embedding_matrix[i] = np.random.normal(0, 0.1, self.embedding_dim)
        
        self.logger.info(f"Found embeddings for {num_words_found}/{min(len(word_index), vocab_size)} words")
        
        # Save embedding matrix
        embedding_matrix_path = os.path.join(self.EMBEDDINGS_DIR, f'{self.embedding_type}_embedding_matrix.npy')
        np.save(embedding_matrix_path, embedding_matrix)
        self.logger.info(f"Embedding matrix saved to {embedding_matrix_path}")
        
        return embedding_matrix
    
    def _initialize_random_embeddings(self, word_index):
        """Initialize a random embedding matrix when pre-trained embeddings aren't available"""
        vocab_size = min(len(word_index) + 1, self.max_features)
        embedding_matrix = np.random.normal(0, 0.1, (vocab_size, self.embedding_dim))
        self.logger.info(f"Initialized random embedding matrix of shape {embedding_matrix.shape}")
        return embedding_matrix
    
    def create_domain_specific_embeddings(self, texts, word_index):
        """
        Create domain-specific embeddings by fine-tuning existing embeddings
        or training from scratch on domain data
        
        Args:
            texts (list): Domain-specific texts for training
            word_index (dict): Dictionary mapping words to indices
            
        Returns:
            numpy.ndarray: Domain-specific embedding matrix
        """
        self.logger.info("Creating domain-specific embeddings...")
        
        # First try to load pre-trained embeddings as a starting point
        try:
            base_matrix = self.load_embedding_matrix(word_index)
            self.logger.info("Using pre-trained embeddings as base for domain adaptation")
        except:
            self.logger.info("Starting domain embeddings from scratch")
            base_matrix = self._initialize_random_embeddings(word_index)
        
        # Create a document-term matrix for learning domain embeddings
        self.logger.info("Building co-occurrence matrix for domain adaptation...")
        
        # Create a basic co-occurrence matrix
        vocab_size = min(len(word_index) + 1, self.max_features)
        window_size = 5
        
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Build co-occurrence matrix (simplified approach)
        cooc_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)
        
        for text in tqdm(texts, desc="Building co-occurrence matrix"):
            words = text.lower().split()
            word_ids = [word_index.get(word, 0) for word in words]
            word_ids = [wid for wid in word_ids if wid < vocab_size]
            
            for i, center_word_id in enumerate(word_ids):
                context_word_ids = word_ids[max(0, i-window_size):i] + word_ids[i+1:min(len(word_ids), i+window_size+1)]
                for context_word_id in context_word_ids:
                    cooc_matrix[center_word_id, context_word_id] += 1
        
        # Apply PPMI weighting (Positive Pointwise Mutual Information)
        self.logger.info("Applying PPMI weighting...")
        row_sums = cooc_matrix.sum(axis=1, keepdims=True)
        col_sums = cooc_matrix.sum(axis=0, keepdims=True)
        total_sum = cooc_matrix.sum()
        
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        col_sums[col_sums == 0] = 1
        
        expected = np.dot(row_sums, col_sums) / total_sum
        mutual_info = np.log(np.maximum(cooc_matrix, 1e-10) / np.maximum(expected, 1e-10))
        ppmi_matrix = np.maximum(mutual_info, 0)
        
        # Reduce dimensionality with SVD
        self.logger.info(f"Reducing dimensionality to {self.embedding_dim} with SVD...")
        svd = TruncatedSVD(n_components=self.embedding_dim, random_state=42)
        domain_embeddings = svd.fit_transform(ppmi_matrix)
        
        # Blend with base embeddings if dimensions match
        if base_matrix.shape == domain_embeddings.shape:
            self.logger.info("Blending domain embeddings with pre-trained embeddings...")
            # 70% pre-trained, 30% domain-specific
            final_embeddings = 0.7 * base_matrix + 0.3 * domain_embeddings
        else:
            final_embeddings = domain_embeddings
            
        # Save domain-specific embeddings
        embedding_matrix_path = os.path.join(self.EMBEDDINGS_DIR, 'domain_specific_embedding_matrix.npy')
        np.save(embedding_matrix_path, final_embeddings)
        self.logger.info(f"Domain-specific embedding matrix saved to {embedding_matrix_path}")
        
        return final_embeddings
    
    def evaluate_embeddings(self, embedding_matrix, word_index, texts=None, labels=None):
        """
        Evaluate the quality of embeddings using various metrics
        
        Args:
            embedding_matrix (numpy.ndarray): Embedding matrix to evaluate
            word_index (dict): Dictionary mapping words to indices
            texts (list): Optional texts for contextual evaluation
            labels (list): Optional labels for task-specific evaluation
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        self.logger.info("Evaluating embedding quality...")
        metrics = {}
        
        # Calculate coverage
        vocab_size = min(len(word_index) + 1, self.max_features)
        non_zero_vectors = np.count_nonzero(np.sum(np.absolute(embedding_matrix), axis=1))
        coverage = non_zero_vectors / vocab_size
        metrics['coverage'] = coverage
        
        # Calculate average vector norm
        vector_norms = np.linalg.norm(embedding_matrix, axis=1)
        avg_norm = np.mean(vector_norms[vector_norms > 0])
        metrics['avg_norm'] = avg_norm
        
        # Calculate variance explained (using PCA)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=5)
        pca.fit(embedding_matrix[1:])  # Skip padding vector
        metrics['variance_explained'] = sum(pca.explained_variance_ratio_)
        
        self.logger.info(f"Embedding metrics: {metrics}")
        return metrics
    
    def prepare_data_for_advanced_training(self, texts, labels, validation_split=0.2,
                                        max_len=500, test_split=0.2, use_cache=True):
        """
        Enhanced data preparation for training with validation split
        
        Args:
            texts (list): List of text strings
            labels (list): Corresponding labels
            validation_split (float): Proportion for validation
            max_len (int): Maximum sequence length
            test_split (float): Proportion for test set
            use_cache (bool): Whether to use cached results
            
        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test, embedding_matrix, max_len)
        """
        # Prepare embeddings with advanced processing
        sequences, embedding_matrix, word_index = self.prepare_advanced_embeddings(texts, use_cache)
        
        # Get vocabulary statistics
        self.logger.info(f"Vocabulary size: {len(word_index)}")
        
        # Calculate appropriate sequence length if not specified
        if max_len is None:
            seq_lengths = [len(seq) for seq in sequences]
            max_len = min(int(np.percentile(seq_lengths, 95)), 500)  # 95th percentile, capped at 500
            self.logger.info(f"Using sequence length: {max_len} (95th percentile)")
        
        # Pad sequences
        X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
        
        # Split into train and temporary test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, labels, test_size=test_split+validation_split, random_state=42, stratify=labels
        )
        
        # Split temporary test into validation and test sets
        relative_val_size = validation_split / (test_split + validation_split)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=relative_val_size, random_state=42, stratify=y_temp
        )
        
        # Log shapes
        self.logger.info(f"Training data: {X_train.shape[0]} samples")
        self.logger.info(f"Validation data: {X_val.shape[0]} samples")
        self.logger.info(f"Testing data: {X_test.shape[0]} samples")
        
        # Evaluate embeddings
        self.evaluate_embeddings(embedding_matrix, word_index, texts, labels)
        
        return X_train, y_train, X_val, y_val, X_test, y_test, embedding_matrix, max_len
    
    def fine_tune_embeddings(self, texts, labels, embedding_matrix, word_index, epochs=5):
        """
        Fine-tune embeddings on domain-specific data using a simple skip-gram model
        
        Args:
            texts (list): Domain texts
            labels (list): Optional labels for supervised fine-tuning
            embedding_matrix (numpy.ndarray): Base embedding matrix to fine-tune
            word_index (dict): Word to index mapping
            epochs (int): Number of training epochs
            
        Returns:
            numpy.ndarray: Fine-tuned embedding matrix
        """
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, Dense, Reshape
        
        self.logger.info("Fine-tuning embeddings on domain data...")
        
        # Prepare skip-gram training data
        samples = []
        labels = []
        window_size = 5
        
        for text in tqdm(texts, desc="Creating skip-gram samples"):
            words = text.lower().split()
            word_ids = [word_index.get(word, 0) for word in words]
            
            for i, target_word_id in enumerate(word_ids):
                for j in range(max(0, i-window_size), min(len(word_ids), i+window_size+1)):
                    if i == j:
                        continue
                    samples.append(target_word_id)
                    labels.append(word_ids[j])
        
        samples = np.array(samples)
        labels = np.array(labels)
        
        # Create skip-gram model
        vocab_size = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]
        
        model = Sequential()
        embedding_layer = Embedding(vocab_size, embed_dim, weights=[embedding_matrix], trainable=True)
        model.add(embedding_layer)
        model.add(Dense(vocab_size, activation='softmax'))
        
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
        
        # Train the model (just a few epochs to fine-tune)
        model.fit(samples, labels, batch_size=512, epochs=epochs, verbose=1)
        
        # Extract the fine-tuned embeddings
        fine_tuned_matrix = embedding_layer.get_weights()[0]
        
        # Save fine-tuned embeddings
        fine_tuned_path = os.path.join(self.EMBEDDINGS_DIR, 'fine_tuned_embedding_matrix.npy')
        np.save(fine_tuned_path, fine_tuned_matrix)
        self.logger.info(f"Fine-tuned embedding matrix saved to {fine_tuned_path}")
        
        return fine_tuned_matrix
