import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, LSTM, Embedding, Input, Dropout, 
    BatchNormalization, Concatenate
)
from tensorflow.keras.models import Model
import numpy as np

class BaseModel:
    def __init__(self, max_features, embedding_dim, max_len, embedding_matrix):
        self.max_features = max_features
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.num_classes = 3  # negative, neutral, positive
        
    def _create_embedding_layer(self):
        """Create an embedding layer initialized with pre-trained embeddings."""
        return Embedding(
            input_dim=self.max_features,
            output_dim=self.embedding_dim,
            weights=[self.embedding_matrix],
            input_length=self.max_len,
            trainable=False,
            name='embedding_layer'
        )

class SimpleLSTMModel(BaseModel):
    """Simple LSTM model with a single LSTM layer."""
    
    def build(self):
        # Input layer
        inputs = Input(shape=(self.max_len,))
        
        # Embedding layer
        x = self._create_embedding_layer()(inputs)
        
        # Dense layer before LSTM
        x = Dense(1000, activation='relu', name='dense_1000')(x)
        
        # LSTM layer
        x = LSTM(100, return_sequences=False, name='lstm_100')(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax', name='output')(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

class DeepLSTMModel(BaseModel):
    """Deep LSTM model with additional dense layers and dropout."""
    
    def build(self):
        # Input layer
        inputs = Input(shape=(self.max_len,))
        
        # Embedding layer
        x = self._create_embedding_layer()(inputs)
        
        # First dense layer
        x = Dense(1000, activation='relu', name='dense_1000')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # LSTM layer
        x = LSTM(100, return_sequences=True, name='lstm_100')(x)
        x = Dropout(0.3)(x)
        
        # Additional dense layers
        x = Dense(128, activation='relu', name='dense_128')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(64, activation='relu', name='dense_64')(x)
        x = BatchNormalization()(x)
        
        # Final LSTM layer
        x = LSTM(50, return_sequences=False, name='lstm_50')(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax', name='output')(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

class StackedLSTMModel(BaseModel):
    """Stacked LSTM model with multiple LSTM layers."""
    
    def build(self):
        # Input layer
        inputs = Input(shape=(self.max_len,))
        
        # Embedding layer
        x = self._create_embedding_layer()(inputs)
        
        # Dense layer
        x = Dense(1000, activation='relu', name='dense_1000')(x)
        x = BatchNormalization()(x)
        
        # First LSTM layer
        x = LSTM(100, return_sequences=True, name='lstm_100')(x)
        x = BatchNormalization()(x)
        
        # Second LSTM layer
        x = LSTM(50, return_sequences=True, name='lstm_50')(x)
        x = BatchNormalization()(x)
        
        # Third LSTM layer
        x = LSTM(25, return_sequences=False, name='lstm_25')(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax', name='output')(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

class EnsembleModel(BaseModel):
    """Ensemble model combining all three architectures."""
    
    def build(self):
        # Input layer
        inputs = Input(shape=(self.max_len,))
        
        # Simple LSTM path
        simple_lstm = self._create_simple_lstm_path(inputs)
        
        # Deep LSTM path
        deep_lstm = self._create_deep_lstm_path(inputs)
        
        # Stacked LSTM path
        stacked_lstm = self._create_stacked_lstm_path(inputs)
        
        # Combine all paths
        combined = Concatenate(name='ensemble_concat')([
            simple_lstm, deep_lstm, stacked_lstm
        ])
        
        # Final dense layers
        x = Dense(64, activation='relu', name='ensemble_dense_64')(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax', name='output')(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_simple_lstm_path(self, inputs):
        x = self._create_embedding_layer()(inputs)
        x = Dense(1000, activation='relu', name='simple_dense_1000')(x)
        x = LSTM(100, return_sequences=False, name='simple_lstm_100')(x)
        return x
    
    def _create_deep_lstm_path(self, inputs):
        x = self._create_embedding_layer()(inputs)
        x = Dense(1000, activation='relu', name='deep_dense_1000')(x)
        x = LSTM(100, return_sequences=True, name='deep_lstm_100')(x)
        x = Dense(128, activation='relu', name='deep_dense_128')(x)
        x = LSTM(50, return_sequences=False, name='deep_lstm_50')(x)
        return x
    
    def _create_stacked_lstm_path(self, inputs):
        x = self._create_embedding_layer()(inputs)
        x = Dense(1000, activation='relu', name='stacked_dense_1000')(x)
        x = LSTM(100, return_sequences=True, name='stacked_lstm_100')(x)
        x = LSTM(50, return_sequences=True, name='stacked_lstm_50')(x)
        x = LSTM(25, return_sequences=False, name='stacked_lstm_25')(x)
        return x 