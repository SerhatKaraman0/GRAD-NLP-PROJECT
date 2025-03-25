from src.common_imports import *  # noqa: F403, F405
from src.nlpmodel import NlpModel
from src.logging_config import *  # noqa: F403, F405
from utils.helper import CONTRACTIONS_DICT, SLANG_DICT  

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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Conv1D, MaxPooling1D, LSTM, Bidirectional, GlobalMaxPooling1D, concatenate, Layer, Reshape, multiply
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

import time
from tensorflow.keras.callbacks import Callback

# To ensure compatibility across environments
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up logging with a more efficient configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AttentionLayer(Layer):
    """
    Attention layer for focusing on important parts of the input sequence.
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                shape=(input_shape[-1], 1),
                                initializer='random_normal',
                                trainable=True)
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # Calculate attention scores
        e = K.tanh(K.dot(x, self.W))
        a = K.softmax(e, axis=1)
        
        # Apply attention weights to input
        output = x * a
        
        return output, a
    
    def compute_output_shape(self, input_shape):
        return input_shape, (input_shape[0], input_shape[1], 1)

class TimeEstimator(Callback):
    """
    Custom callback to estimate training time and show progress within epochs.
    """
    def __init__(self):
        super(TimeEstimator, self).__init__()
        self.epoch_start_time = 0
        self.training_start_time = 0
        self.batch_times = []
        self.total_batches = 0
        self.current_epoch = 0
        self.total_epochs = 0
        
    def on_train_begin(self, logs=None):
        self.training_start_time = time.time()
        print("Starting training...", flush=True)
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.current_epoch = epoch + 1
        self.total_batches = self.params['steps']
        self.total_epochs = self.params['epochs']
        self.batch_times = []
        print(f"\nEpoch {self.current_epoch}/{self.total_epochs}", flush=True)
        
    def on_train_batch_end(self, batch, logs=None):
        if batch == 0:
            # First batch, just record time but don't calculate yet
            self.batch_times.append(time.time())
            return
            
        self.batch_times.append(time.time())
        if len(self.batch_times) >= 2:
            # Calculate time per batch
            batch_time = self.batch_times[-1] - self.batch_times[-2]
            
            # Estimate remaining time for this epoch
            batches_remaining = self.total_batches - (batch + 1)
            est_epoch_remaining = batches_remaining * batch_time
            
            # Calculate progress percentage
            progress = (batch + 1) / self.total_batches * 100
            
            # Format time as hours:minutes:seconds
            h = int(est_epoch_remaining // 3600)
            m = int((est_epoch_remaining % 3600) // 60)
            s = int(est_epoch_remaining % 60)
            
            # Print progress update
            if (batch + 1) % max(1, self.total_batches // 20) == 0 or batch == self.total_batches - 1:  # Update ~20 times per epoch
                print(f"Batch {batch + 1}/{self.total_batches} ({progress:.2f}%) - Est. time remaining for epoch: {h:02d}:{m:02d}:{s:02d}", flush=True)
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        h = int(epoch_time // 3600)
        m = int((epoch_time % 3600) // 60)
        s = int(epoch_time % 60)
        
        # Calculate estimated remaining time for full training
        if self.current_epoch < self.total_epochs:
            epochs_remaining = self.total_epochs - self.current_epoch
            est_total_remaining = epochs_remaining * epoch_time
            h_total = int(est_total_remaining // 3600)
            m_total = int((est_total_remaining % 3600) // 60)
            s_total = int(est_total_remaining % 60)
            
            print(f"Epoch {self.current_epoch}/{self.total_epochs} completed in {h:02d}:{m:02d}:{s:02d} - Est. remaining training time: {h_total:02d}:{m_total:02d}:{s_total:02d}", flush=True)
        else:
            total_time = time.time() - self.training_start_time
            h_total = int(total_time // 3600)
            m_total = int((total_time % 3600) // 60)
            s_total = int(total_time % 60)
            print(f"Training completed in {h_total:02d}:{m_total:02d}:{s_total:02d}", flush=True)

class FeatureEngineering(NlpModel):
    def __init__(self, batch_size=10_000, max_features=7000):
        """
        Initialize paths and load dataset.
        
        Args:
            batch_size: Size of batches for processing
            max_features: Maximum number of features for TF-IDF vectorizer
        """
        super().__init__()
        self.SAVE_DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.PREPROCESSED_DATA_PATH = os.path.join(self.SAVE_DATA_DIR, "PREPROCESSED_Reviews.csv")
        self.batch_size = batch_size
        self.vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=2000,  # Reduced from 7000
            lowercase=True,
            sublinear_tf=True  # Apply sublinear tf scaling (1 + log(tf))
        )
        
        # Defer dataframe loading to when it's needed
        self.df = None
        self.df_size = None
        self.n_batches = None

    def load_data(self):
        """Load data from file and initialize related properties."""
        logger.info(f"Loading data from {self.PREPROCESSED_DATA_PATH}")
        # Use low_memory=False to avoid mixed type inference warnings
        self.df = pd.read_csv(self.PREPROCESSED_DATA_PATH, low_memory=False)
        self.df_size = len(self.df)
        self.n_batches = (self.df_size + self.batch_size - 1) // self.batch_size
        logger.info(f"Data loaded: {self.df_size} records, {self.n_batches} batches")
        
    def preprocess_text(self):
        """Preprocess text data by handling NaNs."""
        if self.df is None:
            self.load_data()
        logger.info("Preprocessing text data")
        # Fill missing values with empty string (more efficient than fillna)
        self.df['Text'] = self.df['Text'].fillna('', inplace=False)

    def _vectorize_tfidf_batch(self, start_idx, end_idx, fit=False):
        """
        Vectorize a batch of data using TF-IDF.
        
        Args:
            start_idx: Starting index of the batch
            end_idx: Ending index of the batch
            fit: Whether to fit the vectorizer on this batch
            
        Returns:
            tuple: (TF-IDF sparse matrix, updated fit flag)
        """
        batch = self.df.iloc[start_idx:end_idx]['Text']
        
        if fit:
            # Only fit on first batch
            batch_tfidf = self.vectorizer.fit_transform(batch)
            return batch_tfidf, False
        else:
            # For subsequent batches, just transform
            batch_tfidf = self.vectorizer.transform(batch)
            return batch_tfidf, False
    
    def process_in_batches(self):
        """
        Process the data in batches to handle large datasets efficiently.
        
        Returns:
            scipy.sparse.csr_matrix: Combined TF-IDF matrix
        """
        self.preprocess_text()
        sparse_matrices = []
        fit = True  # Flag to fit only the first batch
        
        # Use tqdm for progress tracking
        for start_idx in tqdm(range(0, self.df_size, self.batch_size), 
                              desc="Processing batches", 
                              unit="batch"):
            end_idx = min(start_idx + self.batch_size, self.df_size)
            batch_tfidf, fit = self._vectorize_tfidf_batch(start_idx, end_idx, fit)
            
            # Store the sparse matrix
            sparse_matrices.append(batch_tfidf)
            
            # Clean up memory
            gc.collect()
        
        # Combine all sparse matrices efficiently
        logger.info("Combining sparse matrices")
        all_tfidf_matrix = sparse.vstack(sparse_matrices)
        return all_tfidf_matrix
    
    def _vectorize_tfidf(self):
        """Vectorize the entire dataset using batched processing."""
        logger.info("Starting TF-IDF vectorization")
        return self.process_in_batches()

    def build_model(self, input_dim):
        """
        Build a triple-path model with enhanced feature extraction.
        
        Args:
            input_dim: Dimension of the input features
                
        Returns:
            tf.keras.models.Model: Compiled Keras model
        """
        logger.info(f"Building triple-path model with input dimension {input_dim}")
        
        # Common input layer
        input_layer = Input(shape=(input_dim,))
        
        # Path 1: Deep dense layers for complex patterns
        path1 = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(input_layer)
        path1 = BatchNormalization()(path1)
        path1 = Dropout(0.5)(path1)
        
        path1 = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(path1)
        path1 = BatchNormalization()(path1)
        path1 = Dropout(0.4)(path1) 
        
        path1 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(path1)
        path1 = BatchNormalization()(path1)
        path1 = Dropout(0.3)(path1)

        path1 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(path1)
        path1 = BatchNormalization()(path1)
        path1 = Dropout(0.2)(path1)
        
        # Path 2: Wider, shallower network for broader feature detection
        path2 = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(input_layer)
        path2 = BatchNormalization()(path2)
        path2 = Dropout(0.5)(path2)
        
        path2 = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(path2)
        path2 = BatchNormalization()(path2)
        path2 = Dropout(0.4)(path2)
        
        # Path 3: Specialized for sentiment detection with attention mechanism
        path3 = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(input_layer)
        path3 = BatchNormalization()(path3)
        path3 = Dropout(0.3)(path3)
        
        # Add attention mechanism correctly using tf.keras.layers.multiply
        attention_weights = Dense(256, activation='softmax', name='attention_weights')(path3)
        path3_attention = tf.keras.layers.multiply([path3, attention_weights])
        
        # Combine all three paths
        combined = concatenate([path1, path2, path3_attention])
        
        # Common layers after combining
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(combined)
        x = Dropout(0.3)(x)
        
        # Output layer
        output = Dense(1, activation='linear')(x)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output)
        
        # Custom accuracy metric with improved stability
        def accuracy_metric(y_true, y_pred):
            y_true_rounded = tf.round(y_true)
            y_pred_rounded = tf.round(y_pred)
            y_true_rounded = tf.cast(y_true_rounded, tf.int32)
            y_pred_rounded = tf.cast(y_pred_rounded, tf.int32)
            # Clip predictions to valid range
            y_pred_rounded = tf.clip_by_value(y_pred_rounded, 1, 5)
            return tf.reduce_mean(tf.cast(tf.equal(y_true_rounded, y_pred_rounded), tf.float32))
        
        # Compile model with Adam optimizer and gradient clipping
        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', accuracy_metric]
        )
        
        logger.info(f"Model built with {model.count_params()} parameters")
        return model
    
    # Add a classification head that handles ratings as categories
    def build_dual_head_model(self, input_dim):
        """Build a model with both regression and classification heads."""
        # Common input and initial layers
        input_layer = Input(shape=(input_dim,))
        
        shared = Dense(512, activation='relu')(input_layer)
        shared = Dropout(0.5)(shared)
        shared = Dense(256, activation='relu')(shared)
        shared = Dropout(0.3)(shared)
        
        # Regression head
        regression_output = Dense(1, name='regression')(shared)
        
        # Classification head (5 classes for ratings 1-5)
        classification_output = Dense(5, activation='softmax', name='classification')(shared)
        
        # Create model with multiple outputs
        model = Model(inputs=input_layer, outputs=[regression_output, classification_output])
        
        # Define custom metrics
        def regression_accuracy(y_true, y_pred):
            y_true_rounded = tf.round(y_true)
            y_pred_rounded = tf.round(y_pred)
            return tf.reduce_mean(tf.cast(tf.equal(y_true_rounded, y_pred_rounded), tf.float32))
        
        # Compile with multiple loss functions
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'regression': 'mse',
                'classification': 'sparse_categorical_crossentropy'
            },
            metrics={
                'regression': ['mae', regression_accuracy],
                'classification': 'accuracy'
            },
            loss_weights={
                'regression': 0.3, 
                'classification': 0.7
            }
        )
        
        return model
    
    def train_dual_head_model(self, X, y):
        """
        Train a dual-output model that handles both regression and classification.
        
        Args:
            X: Training features
            y: Target values (regression scores)
                    
        Returns:
            tf.keras.models.Model: Trained model
        """
        # Build the dual-head model
        model = self.build_dual_output_model(X.shape[1])
        
        # Convert regression labels to classification labels (0-4 for classes 1-5)
        y_classes = np.round(y).astype(int) - 1  # Subtract 1 to make 0-indexed
        y_classes = np.clip(y_classes, 0, 4)  # Ensure valid range
        
        # Create class weights to handle class imbalance
        class_counts = np.bincount(y_classes)
        total_samples = len(y_classes)
        class_weights = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}
        
        # Print class distribution
        logger.info(f"Class distribution: {class_counts}")
        logger.info(f"Class weights: {class_weights}")
        
        # Create validation split with stratification
        X_train, X_val, y_train, y_val, y_classes_train, y_classes_val = train_test_split(
            X, y, y_classes, test_size=0.2, random_state=42,
            stratify=y_classes
        )
        
        # Enhanced callbacks for training
        callbacks = [
            EarlyStopping(
                monitor='val_classification_accuracy',  # Focus on classification accuracy
                patience=15,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ModelCheckpoint(
                os.path.join(self.SAVE_DATA_DIR, "dual_model_best.h5"),
                monitor='val_classification_accuracy',
                save_best_weights_only=True,
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_classification_accuracy',
                factor=0.5,
                patience=7,
                min_lr=0.00001,
                mode='max',
                verbose=1
            ),
            TimeEstimator()
        ]
        
        # Train the model with both outputs
        logger.info("Starting dual-head model training")
        history = model.fit(
            X_train,
            {'regression': y_train, 'classification': y_classes_train},
            epochs=50,  # Train for longer
            batch_size=128,
            validation_data=(X_val, {'regression': y_val, 'classification': y_classes_val}),
            callbacks=callbacks,
            class_weight={'classification': class_weights},  # Apply class weights
            verbose=1
        )
        
        logger.info("Dual-head model training completed")
        
        # Create additional visualizations for dual training
        self.visualize_dual_training_history(history)
        
        return model
    
    def train_with_cross_validation(self, X, y, n_splits=5):
        """Train with cross-validation to get more reliable performance estimates."""
        from sklearn.model_selection import StratifiedKFold
        
        # Initialize arrays to store results
        val_accuracies = []
        val_f1_scores = []
        
        # Initialize StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Convert to integer classes for stratification
        y_classes = np.round(y).astype(int)
        
        # Loop over folds
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_classes)):
            logger.info(f"Training on fold {fold+1}/{n_splits}")
            
            # Split data
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Build and train model
            model = self.build_model(X.shape[1])
            
            # Train model
            model.fit(
                X_train_fold, 
                y_train_fold,
                epochs=30,
                batch_size=128,
                validation_data=(X_val_fold, y_val_fold),
                callbacks=[
                    EarlyStopping(monitor='val_accuracy_metric', patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_accuracy_metric', factor=0.5, patience=5)
                ],
                verbose=1
            )
            
            # Evaluate model
            eval_results = model.evaluate(X_val_fold, y_val_fold, verbose=0)
            
            # Get predictions and calculate metrics
            y_pred = model.predict(X_val_fold).flatten()
            y_pred_classes = np.round(y_pred).astype(int)
            y_val_classes = np.round(y_val_fold).astype(int)
            
            accuracy = accuracy_score(y_val_classes, y_pred_classes)
            f1 = f1_score(y_val_classes, y_pred_classes, average='weighted')
            
            val_accuracies.append(accuracy)
            val_f1_scores.append(f1)
            
            logger.info(f"Fold {fold+1} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        
        # Calculate and log average results
        avg_accuracy = np.mean(val_accuracies)
        avg_f1 = np.mean(val_f1_scores)
        
        logger.info(f"Cross-validation results:")
        logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
        logger.info(f"Average F1-Score: {avg_f1:.4f}")
        
        # Train a final model on all data
        final_model = self.build_model(X.shape[1])
        final_model.fit(
            X, y,
            epochs=30,
            batch_size=128,
            validation_split=0.1,  # Small validation set for monitoring
            callbacks=[
                EarlyStopping(monitor='val_accuracy_metric', patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_accuracy_metric', factor=0.5, patience=5)
            ],
            verbose=1
        )
        
        return final_model, avg_accuracy, avg_f1
    
    def build_stacked_ensemble(self):
        """Create a stacked ensemble of multiple model types."""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import StackingClassifier
        
        # Define base classifiers
        base_classifiers = [
            ('rf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('svc', SVC(probability=True, random_state=42)),
        ]
        
        # Define meta-classifier
        meta_classifier = LogisticRegression(C=10.0, max_iter=1000)
        
        # Create stacked ensemble
        stacked_model = StackingClassifier(
            estimators=base_classifiers,
            final_estimator=meta_classifier,
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        
        return stacked_model
    
    def build_dual_output_model(self, input_dim):
        """
        Build a model with both regression and classification outputs.
        
        Args:
            input_dim: Dimension of input features
            
        Returns:
            keras.Model: Compiled model
        """
        # Input layer
        input_layer = Input(shape=(input_dim,))
        
        # Shared backbone
        backbone = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(input_layer)
        backbone = BatchNormalization()(backbone)
        backbone = Dropout(0.5)(backbone)
        
        backbone = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(backbone)
        backbone = BatchNormalization()(backbone)
        backbone = Dropout(0.4)(backbone)
        
        # Regression head
        reg_branch = Dense(128, activation='relu')(backbone)
        reg_branch = Dropout(0.3)(reg_branch)
        reg_output = Dense(1, name='regression')(reg_branch)
        
        # Classification head (5 classes for ratings 1-5)
        cls_branch = Dense(128, activation='relu')(backbone)
        cls_branch = Dropout(0.3)(cls_branch)
        cls_output = Dense(5, activation='softmax', name='classification')(cls_branch)
        
        # Create model with two outputs
        model = Model(inputs=input_layer, outputs=[reg_output, cls_output])
        
        # Custom metrics
        def accuracy_metric(y_true, y_pred):
            y_true_rounded = tf.round(y_true)
            y_pred_rounded = tf.round(y_pred)
            return tf.reduce_mean(tf.cast(tf.equal(y_true_rounded, y_pred_rounded), tf.float32))
        
        # Compile model with multiple losses
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'regression': 'mse',
                'classification': 'sparse_categorical_crossentropy'
            },
            metrics={
                'regression': ['mae', accuracy_metric],
                'classification': 'accuracy'
            },
            loss_weights={
                'regression': 0.3, 
                'classification': 0.7  # Give more weight to classification task
            }
        )
        
        return model
    
    def focal_loss(self, gamma=2.0, alpha=0.25):
        """
        Implement focal loss for better handling of imbalanced classes.
        
        Args:
            gamma: Focusing parameter
            alpha: Class weight parameter
        
        Returns:
            Loss function
        """
        def loss_function(y_true, y_pred):
            # Convert regression to classification
            y_true_int = tf.cast(tf.round(y_true), tf.int32)
            y_pred_prob = tf.nn.sigmoid(y_pred)
            
            # One-hot encode the targets
            num_classes = 5
            y_true_one_hot = tf.one_hot(y_true_int - 1, depth=num_classes)
            
            # Calculate focal loss
            alpha_factor = tf.ones_like(y_true_one_hot) * alpha
            alpha_factor = tf.where(tf.equal(y_true_one_hot, 1), alpha_factor, 1 - alpha_factor)
            
            focal_weight = tf.where(tf.equal(y_true_one_hot, 1), 
                                1 - y_pred_prob, 
                                y_pred_prob)
            focal_weight = alpha_factor * tf.pow(focal_weight, gamma)
            
            cls_loss = focal_weight * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=y_true_one_hot, 
                logits=y_pred)
            
            return tf.reduce_sum(cls_loss)
        
        return loss_function
    
    def build_lstm_model(self, vocab_size, embedding_dim=100, max_seq_length=200):
        """Build an LSTM-based model with attention for better sequence processing."""
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        # Tokenize texts
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(self.df['Text'])
        sequences = tokenizer.texts_to_sequences(self.df['Text'])
        
        # Save tokenizer for later use
        import pickle
        with open(os.path.join(self.SAVE_DATA_DIR, 'tokenizer.pickle'), 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Pad sequences
        data = pad_sequences(sequences, maxlen=max_seq_length)
        
        # Create embedding matrix
        embedding_matrix = self.create_embedding_matrix(tokenizer.word_index, embedding_dim)
        
        # Define model
        input_layer = Input(shape=(max_seq_length,))
        
        # Embedding layer with pre-trained weights
        embedding_layer = Embedding(
            input_dim=len(tokenizer.word_index) + 1,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_seq_length,
            trainable=False
        )(input_layer)
        
        # Bidirectional LSTM layer
        lstm_layer = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
        
        # Add attention layer
        attention = Dense(1, activation='tanh')(lstm_layer)
        attention = tf.squeeze(attention, axis=-1)
        attention_weights = tf.nn.softmax(attention)
        context = tf.expand_dims(attention_weights, axis=-1) * lstm_layer
        context = tf.reduce_sum(context, axis=1)
        
        # Add dense layers
        x = Dense(128, activation='relu')(context)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        # Output layer
        output = Dense(1, activation='linear')(x)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', self.accuracy_metric]
        )
        
        return model, data
    
    def add_sentiment_lexicon_features(self):
        """Add features from sentiment lexicons."""
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        
        # Download VADER lexicon if needed
        import nltk
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon')
        
        sid = SentimentIntensityAnalyzer()
        
        # Create numpy array for features
        vader_features = np.zeros((self.df_size, 4))
        
        # Process in batches
        for start_idx in tqdm(range(0, self.df_size, self.batch_size), desc="Adding VADER features"):
            end_idx = min(start_idx + self.batch_size, self.df_size)
            batch = self.df.iloc[start_idx:end_idx]
            
            for i, text in enumerate(batch["Text"]):
                text = str(text)
                scores = sid.polarity_scores(text)
                
                # Store the scores
                vader_features[start_idx + i, 0] = scores['neg']
                vader_features[start_idx + i, 1] = scores['neu']
                vader_features[start_idx + i, 2] = scores['pos']
                vader_features[start_idx + i, 3] = scores['compound']
        
        return vader_features
    
    def create_embedding_matrix(self, word_index, embedding_dim=100):
        """Create an embedding matrix from pre-trained GloVe embeddings."""
        import numpy as np
        import os
        
        # Download GloVe embeddings if not present
        glove_path = os.path.join(self.SAVE_DATA_DIR, "glove.6B.100d.txt")
        if not os.path.exists(glove_path):
            import urllib.request
            import zipfile
            
            logger.info("Downloading GloVe embeddings...")
            url = "http://nlp.stanford.edu/data/glove.6B.zip"
            zip_path = os.path.join(self.SAVE_DATA_DIR, "glove.6B.zip")
            urllib.request.urlretrieve(url, zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.SAVE_DATA_DIR)
            
            os.remove(zip_path)
        
        # Load the GloVe embeddings
        logger.info("Loading GloVe embeddings...")
        embeddings_index = {}
        with open(glove_path, encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading GloVe"):
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        
        logger.info(f"Found {len(embeddings_index)} word vectors.")
        
        # Create the embedding matrix
        embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        
        return embedding_matrix

    def train_model(self, X, y):
        """
        Train the model with improved training settings.
        
        Args:
            X: Training features
            y: Target values
                    
        Returns:
            tf.keras.models.Model: Trained model
        """
        # Build the model
        model = self.build_model(X.shape[1])
        
        # Enhanced callbacks for training
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy_metric',
                patience=15,  # Increased from 10
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ModelCheckpoint(
                os.path.join(self.SAVE_DATA_DIR, "sentiment_model_best.h5"),
                monitor='val_accuracy_metric',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy_metric',
                factor=0.5,
                patience=7,  # Increased from 5
                min_lr=0.00001,
                mode='max',
                verbose=1
            ),
            TimeEstimator()  # Custom time estimator callback
        ]
        
        # Train the model with progress bar
        logger.info("Starting model training")
        history = model.fit(
            X, y,
            epochs=30,  # Increased from 20
            batch_size=128,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Model training completed")
        
        # Visualize training history
        self.visualize_training_history(history)
        
        return model

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the model on the test set with comprehensive metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target values
                    
        Returns:
            tuple: (loss, mean absolute error, predicted values)
        """
        logger.info("Evaluating model on test data")
        # Get evaluation metrics
        evaluation = model.evaluate(X_test, y_test, verbose=0)
        loss = evaluation[0]
        mae = evaluation[1]
        accuracy = evaluation[2]
        
        logger.info(f"Model evaluation: Loss={loss:.4f}, MAE={mae:.4f}, Accuracy={accuracy:.4f}")
        
        # Get predictions
        y_pred = model.predict(X_test, verbose=0).flatten()
        
        # Calculate additional metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # For classification metrics
        y_test_rounded = np.round(y_test).astype(int)
        y_pred_rounded = np.round(y_pred).astype(int)
        y_pred_rounded = np.clip(y_pred_rounded, 1, 5)  # Ensure valid range
        
        # Classification metrics
        acc = accuracy_score(y_test_rounded, y_pred_rounded)
        f1 = f1_score(y_test_rounded, y_pred_rounded, average='weighted')
        
        # Log all metrics
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"R²: {r2:.4f}")
        logger.info(f"Classification Accuracy: {acc:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        # Create confusion matrix visualization
        cm = confusion_matrix(y_test_rounded, y_pred_rounded)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(1, 6), yticklabels=range(1, 6))
        plt.xlabel('Predicted Rating')
        plt.ylabel('True Rating')
        plt.title('Confusion Matrix')
        
        # Save confusion matrix
        cm_path = os.path.join(self.SAVE_DATA_DIR, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        logger.info(f"Confusion matrix saved to {cm_path}")
        
        return loss, mae, y_pred

    def visualize_training_history(self, history):
        """
        Visualize the model training history.
            
        Args:
            history: Keras history object from model.fit()
        """
        metrics_dir = os.path.join(self.SAVE_DATA_DIR, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
            
        # Convert history to DataFrame
        history_df = pd.DataFrame(history.history)
            
        # Save history to CSV
        history_df.to_csv(os.path.join(metrics_dir, 'training_history.csv'), index=False)
            
        # Plot training & validation loss
        plt.figure(figsize=(12, 5))
            
        plt.subplot(1, 2, 1)
        plt.plot(history_df['loss'], label='Training Loss')
        plt.plot(history_df['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
            
        plt.subplot(1, 2, 2)
        plt.plot(history_df['mae'], label='Training MAE')
        plt.plot(history_df['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
            
        # Learning rate plot if available
        if 'lr' in history_df.columns:
            plt.figure(figsize=(10, 4))
            plt.plot(history_df['lr'])
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.savefig(os.path.join(metrics_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training history visualizations saved to {metrics_dir}")

    def visualize_dual_training_history(self, history):
        """
        Visualize the training history for a dual-output model.
            
        Args:
            history: Keras history object from model.fit()
        """
        metrics_dir = os.path.join(self.SAVE_DATA_DIR, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
            
        # Convert history to DataFrame
        history_df = pd.DataFrame(history.history)
            
        # Save history to CSV
        history_df.to_csv(os.path.join(metrics_dir, 'dual_training_history.csv'), index=False)
            
        # Create a 2x2 grid of plots
        plt.figure(figsize=(15, 12))
            
        # Plot 1: Regression Loss
        plt.subplot(2, 2, 1)
        plt.plot(history_df['regression_loss'], label='Training')
        plt.plot(history_df['val_regression_loss'], label='Validation')
        plt.title('Regression Loss (MSE)')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.legend()
            
        # Plot 2: Classification Loss
        plt.subplot(2, 2, 2)
        plt.plot(history_df['classification_loss'], label='Training')
        plt.plot(history_df['val_classification_loss'], label='Validation')
        plt.title('Classification Loss (Cross-Entropy)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
            
        # Plot 3: Regression Accuracy
        plt.subplot(2, 2, 3)
        plt.plot(history_df['regression_regression_accuracy'], label='Training')
        plt.plot(history_df['val_regression_regression_accuracy'], label='Validation')
        plt.title('Regression Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
            
        # Plot 4: Classification Accuracy
        plt.subplot(2, 2, 4)
        plt.plot(history_df['classification_accuracy'], label='Training')
        plt.plot(history_df['val_classification_accuracy'], label='Validation')
        plt.title('Classification Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, 'dual_training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
            
        logger.info(f"Dual training history visualizations saved to {metrics_dir}")

    def create_additional_features(self):
        """Add domain-specific features to improve model performance."""
        import nltk
        from textblob import TextBlob
        
        # Download necessary NLTK resources if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        logger.info("Creating additional language features")
        
        # Process in batches to avoid memory issues
        additional_features = []
        
        for start_idx in tqdm(range(0, self.df_size, self.batch_size), desc="Adding features"):
            end_idx = min(start_idx + self.batch_size, self.df_size)
            batch = self.df.iloc[start_idx:end_idx]
            
            # Create feature arrays for this batch
            batch_features = np.zeros((len(batch), 8))  # 8 linguistic features
            
            for i, text in enumerate(batch["Text"]):
                text = str(text)
                words = nltk.word_tokenize(text)
                sentences = nltk.sent_tokenize(text)
                
                # TextBlob sentiment
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Text statistics
                avg_word_len = np.mean([len(w) for w in words]) if words else 0
                avg_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0
                
                # Sentiment indicators
                exclamation_count = text.count('!')
                question_count = text.count('?')
                
                # Count uppercase words (potential intensity)
                uppercase_words = sum(1 for word in words if word.isupper() and len(word) > 1)
                
                # Count positive and negative words
                pos_words = sum(1 for s in blob.sentences if s.sentiment.polarity > 0.5)
                neg_words = sum(1 for s in blob.sentences if s.sentiment.polarity < -0.5)
                
                # Store features
                batch_features[i] = [
                    polarity, subjectivity, avg_word_len, avg_sent_len,
                    exclamation_count, question_count, pos_words, neg_words
                ]
                
            additional_features.append(batch_features)
            gc.collect()
        
        # Combine all batches
        all_features = np.vstack(additional_features)
        
        # Return feature matrix
        return all_features
    
    def build_ensemble_model(self, X_train, y_train):
        """Build and train an ensemble model combining multiple models."""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear.model import Ridge
        from sklearn.svm import SVR
        from sklearn.ensemble import VotingRegressor
        
        logger.info("Training ensemble model")
        
        # Convert sparse to dense if needed
        if sparse.issparse(X_train):
            X_train_dense = X_train.toarray()
        else:
            X_train_dense = X_train
        
        # Initialize base models
        models = [
            ('ridge', Ridge(alpha=1.0)),
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=30, random_state=42, n_jobs=-1)),
            ('gbr', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
        ]
        
        # Create and train ensemble
        ensemble = VotingRegressor(models)
        ensemble.fit(X_train_dense, y_train)
        
        return ensemble

    def build_bert_model(self):
        """Build a BERT-based model for sentiment analysis."""
        import tensorflow_hub as hub
        import tensorflow_text as text
        
        logger.info("Building BERT-based model")
        
        # Load BERT encoder from TensorFlow Hub
        bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2")
        
        # Text input for BERT
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input')
        
        # Preprocess text for BERT
        preprocessed_text = bert_preprocess(text_input)
        
        # Get BERT embeddings
        embeddings = bert_encoder(preprocessed_text)['pooled_output']
        
        # Add dropout and dense layers
        x = tf.keras.layers.Dropout(0.1)(embeddings)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(1)(x)
        
        # Create the model
        model = tf.keras.Model(inputs=text_input, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model

    def build_interactive_dashboard(self, model, X_test, y_test, y_pred):
        """
        Build an interactive HTML dashboard of model metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            y_pred: Predicted values
        """
        metrics_dir = os.path.join(self.SAVE_DATA_DIR, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        mse = mean_squared_error(y_test, y_pred)
        
        # For classification-like metrics, round predictions
        y_test_rounded = np.round(y_test).astype(int)
        y_pred_rounded = np.round(y_pred).astype(int)
        
        # Cap to valid range
        y_pred_rounded = np.clip(y_pred_rounded, 1, 5)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test_rounded, y_pred_rounded)
        
        # Create a confusion matrix
        conf_matrix = confusion_matrix(y_test_rounded, y_pred_rounded)
        
        # Function to generate base64 image from figure
        def fig_to_base64(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            return img_str
        
        # Create figures and convert to base64
        
        # 1. Actual vs Predicted
        fig1 = plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.xlabel('Actual Ratings')
        plt.ylabel('Predicted Ratings')
        plt.title('Actual vs Predicted Ratings')
        plt.grid(True, alpha=0.3)
        actual_vs_pred_img = fig_to_base64(fig1)
        plt.close(fig1)
        
        # 2. Error distribution
        errors = y_pred - y_test
        fig2 = plt.figure(figsize=(8, 6))
        sns.histplot(errors, kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.title(f'Error Distribution (MAE: {mae:.4f})')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.grid(True, alpha=0.3)
        error_dist_img = fig_to_base64(fig2)
        plt.close(fig2)
        
        # 3. Confusion Matrix
        fig3 = plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=range(1, 6), yticklabels=range(1, 6))
        plt.xlabel('Predicted Rating')
        plt.ylabel('Actual Rating')
        plt.title('Confusion Matrix (Rounded Ratings)')
        conf_matrix_img = fig_to_base64(fig3)
        plt.close(fig3)
        
        # Create HTML dashboard
        html_content = f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Sentiment Analysis Model Dashboard</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .header {{
                    background-color: #4a86e8;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .metric-box {{
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    padding: 15px;
                    margin-bottom: 15px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metrics-container {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-between;
                    margin-bottom: 20px;
                }}
                .metric-item {{
                    width: 22%;
                    text-align: center;
                    background-color: #e8f4f8;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #4a86e8;
                }}
                .charts-container {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-between;
                }}
                .chart-box {{
                    width: 48%;
                    margin-bottom: 20px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    padding: 15px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .chart-box img {{
                    width: 100%;
                    height: auto;
                }}
                .full-width {{
                    width: 100%;
                }}
                h2 {{
                    color: #4a86e8;
                }}
                @media (max-width: 768px) {{
                    .metric-item {{
                        width: 48%;
                        margin-bottom: 15px;
                    }}
                    .chart-box {{
                        width: 100%;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Sentiment Analysis Model Dashboard</h1>
                    <p>Model performance metrics and visualizations</p>
                </div>
                
                <div class="metric-box">
                    <h2>Key Performance Metrics</h2>
                    <div class="metrics-container">
                        <div class="metric-item" style="background-color: #e8f4f8; border: 2px solid #4a86e8;">
                            <h3>Accuracy</h3>
                            <div class="metric-value" style="color: #2b579a; font-size: 32px;">{accuracy:.4f}</div>
                            <p>Classification Accuracy</p>
                        </div>
                        <div class="metric-item">
                            <h3>MAE</h3>
                            <div class="metric-value">{mae:.4f}</div>
                            <p>Mean Absolute Error</p>
                        </div>
                        <div class="metric-item">
                            <h3>MSE</h3>
                            <div class="metric-value">{mse:.4f}</div>
                            <p>Mean Squared Error</p>
                        </div>
                        <div class="metric-item">
                            <h3>RMSE</h3>
                            <div class="metric-value">{rmse:.4f}</div>
                            <p>Root Mean Squared Error</p>
                        </div>
                        <div class="metric-item">
                            <h3>R²</h3>
                            <div class="metric-value">{r2:.4f}</div>
                            <p>Coefficient of Determination</p>
                        </div>
                    </div>
                </div>
                
                <div class="charts-container">
                    <div class="chart-box">
                        <h2>Actual vs Predicted Ratings</h2>
                        <img src="data:image/png;base64,{actual_vs_pred_img}" alt="Actual vs Predicted Plot">
                    </div>
                    <div class="chart-box">
                        <h2>Error Distribution</h2>
                        <img src="data:image/png;base64,{error_dist_img}" alt="Error Distribution">
                    </div>
                    <div class="chart-box full-width">
                        <h2>Confusion Matrix (Rounded Ratings)</h2>
                        <img src="data:image/png;base64,{conf_matrix_img}" alt="Confusion Matrix">
                    </div>
                </div>
                
                <div class="metric-box">
                    <h2>Model Information</h2>
                    <p><strong>Features:</strong> {model.input_shape[1]} TF-IDF features</p>
                    <p><strong>Architecture:</strong> {model.input_shape[1]} → 1000 → 100 → 128 → 64 → 1</p>
                    <p><strong>Test Set Size:</strong> {len(y_test)} samples</p>
                    <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        '''
        
        # Save the HTML dashboard
        dashboard_path = os.path.join(metrics_dir, 'model_dashboard.html')
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Interactive dashboard saved to {dashboard_path}")
        return dashboard_path

    def save_to_parquet(self, df, output_path):
        """
        Save the processed DataFrame to Parquet with gzip compression.
        
        Args:
            df: DataFrame to save
            output_path: Path to save the file (without extension)
        """
        logger.info(f"Saving DataFrame to {output_path}.parquet.gz")
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to Parquet format with gzip compression
        df.to_parquet(f"{output_path}.parquet.gz", compression="gzip")
        
        logger.info(f"Data saved to {output_path}.parquet.gz")

    def load_and_process_data(self):
        """
        Load and process data, returning train/test splits.
        
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        if self.df is None:
            self.load_data()
        
        # Generate TF-IDF features
        X_sparse = self._vectorize_tfidf()
        y = self.df["Score"].values
        
        logger.info(f"TF-IDF matrix shape: {X_sparse.shape}")
        
        # Split the data into training and test sets
        logger.info("Splitting data into train and test sets")
        X_train, X_test, y_train, y_test = train_test_split(
            X_sparse, y, 
            test_size=0.2, 
            random_state=42,
            stratify=None  # Can use stratify=pd.qcut(y, 5, duplicates='drop') if needed
        )
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Check if we need to convert to dense
        if isinstance(X_train, sparse.spmatrix) and tf.config.list_physical_devices('GPU'):
            logger.info("Converting sparse matrices to dense for GPU training")
            X_train_processed = X_train.toarray()
            X_test_processed = X_test.toarray()
        else:
            # Keep as sparse for CPU training or if already dense
            X_train_processed = X_train
            X_test_processed = X_test
            
        return X_train_processed, y_train, X_test_processed, y_test
    
    def save_feature_importance(self, model, output_path=None):
        """
        Save feature importance scores from the model.
        
        Args:
            model: Trained model
            output_path: Path to save the scores (optional)
        """
        if output_path is None:
            output_path = os.path.join(self.SAVE_DATA_DIR, "feature_importance.csv")
            
        logger.info("Extracting feature importance")
        
        # Get feature names from vectorizer
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get weights from the first layer
        weights = model.layers[0].get_weights()[0]
        
        # Create a DataFrame with feature names and weights
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(weights.mean(axis=1))
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Save to CSV
        importance_df.to_csv(output_path, index=False)
        logger.info(f"Feature importance saved to {output_path}")
        
        return importance_df.head(20)  # Return top 20 features for inspection

    def create_advanced_features(self):
        """Create advanced NLP features to improve sentiment analysis."""
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk
        
        # Download necessary NLTK resources
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon')
            
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Initialize VADER sentiment analyzer
        sid = SentimentIntensityAnalyzer()
        
        # Create feature array with 14 columns
        feature_array = np.zeros((self.df_size, 14))
        
        # Process in batches
        for start_idx in tqdm(range(0, self.df_size, self.batch_size), desc="Creating advanced features"):
            end_idx = min(start_idx + self.batch_size, self.df_size)
            batch = self.df.iloc[start_idx:end_idx]
            
            for i, (_, row) in enumerate(batch.iterrows()):
                text = str(row['Text'])
                
                # 1-4: VADER sentiment features
                vader_scores = sid.polarity_scores(text)
                feature_array[start_idx + i, 0] = vader_scores['neg']
                feature_array[start_idx + i, 1] = vader_scores['neu']
                feature_array[start_idx + i, 2] = vader_scores['pos']
                feature_array[start_idx + i, 3] = vader_scores['compound']
                
                # 5: Text length
                feature_array[start_idx + i, 4] = len(text)
                
                # 6: Word count
                words = nltk.word_tokenize(text)
                feature_array[start_idx + i, 5] = len(words)
                
                # 7: Average word length
                if words:
                    feature_array[start_idx + i, 6] = sum(len(w) for w in words) / len(words)
                
                # 8: Sentence count
                sentences = nltk.sent_tokenize(text)
                feature_array[start_idx + i, 7] = len(sentences)
                
                # 9: Average sentence length
                if sentences:
                    sent_lengths = [len(nltk.word_tokenize(s)) for s in sentences]
                    feature_array[start_idx + i, 8] = sum(sent_lengths) / len(sentences)
                
                # 10-14: Punctuation counts
                feature_array[start_idx + i, 9] = text.count('!')   # Exclamation points
                feature_array[start_idx + i, 10] = text.count('?')  # Question marks
                feature_array[start_idx + i, 11] = sum(1 for c in text if c.isupper()) / max(1, len(text))  # % uppercase
                feature_array[start_idx + i, 12] = text.count('.') + text.count('!') + text.count('?')  # Sentence ends
                feature_array[start_idx + i, 13] = len(re.findall(r'[A-Z]{2,}', text))  # Count of all-caps words
        
        logger.info(f"Created {feature_array.shape[1]} advanced features")
        return feature_array


def main():
    """Main execution function with dual-head model."""
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
    model = FeatureEngineering(batch_size=10_000, max_features=7000)
    
    try:
        # Load data and process TF-IDF
        X_train, y_train, X_test, y_test = model.load_and_process_data()
        
        # Train the model
        trained_model = model.train_model(X_train, y_train)
        
        # Evaluate the model with comprehensive metrics
        loss, mae, y_pred = model.evaluate_model(trained_model, X_test, y_test)

        
        # Create interactive dashboard
        dashboard_path = model.build_interactive_dashboard(trained_model, X_test, y_test, y_pred)
        logger.info(f"Interactive dashboard created at: {dashboard_path}")
        
        # Save the model
        model_path = os.path.join(model.SAVE_DATA_DIR, "sentiment_model_final.h5")
        trained_model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save feature importance
        top_features = model.save_feature_importance(trained_model)
        logger.info(f"Top features:\n{top_features}")
        
    except Exception as e:
        logger.error(f"Error in processing: {e}", exc_info=True)


if __name__ == "__main__":
    main()