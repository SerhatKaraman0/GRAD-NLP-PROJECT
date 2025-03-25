import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Embedding, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class BaseModel:
    """Base class for all sentiment analysis models."""
    def __init__(self, input_dim, embedding_dim=32):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
    def _accuracy_metric(self, y_true, y_pred):
        """Custom accuracy metric for sentiment analysis."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_rounded = tf.cast(tf.round(y_true), tf.float32)
        y_pred_rounded = tf.cast(tf.round(tf.clip_by_value(y_pred, 1, 5)), tf.float32)
        return tf.reduce_mean(tf.cast(tf.equal(y_true_rounded, y_pred_rounded), tf.float32))
    
    def get_callbacks(self):
        """Get common training callbacks."""
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.0001,
                mode='min'
            )
        ]

class SimpleLSTMModel(BaseModel):
    """First model architecture with simple LSTM."""
    def build(self):
        # Input layer
        input_layer = Input(shape=(self.input_dim,), name='embedding_1_input')
        
        # Embedding layer
        x = Dense(1000, activation='relu', name='embedding_1_dense')(input_layer)
        x = Embedding(input_dim=1000, output_dim=self.embedding_dim, name='embedding_1')(x)
        
        # LSTM layer
        x = LSTM(100, name='lstm_1')(x)
        
        # Output layer
        output = Dense(1, name='dense_1')(x)
        
        # Create and compile model
        model = Model(inputs=input_layer, outputs=output, name='simple_lstm_model')
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', self._accuracy_metric]
        )
        
        return model

class DeepLSTMModel(BaseModel):
    """Second model architecture with LSTM and multiple dense layers."""
    def build(self):
        # Input layer
        input_layer = Input(shape=(self.input_dim,), name='embedding_2_input')
        
        # Embedding layer
        x = Dense(1000, activation='relu', name='embedding_2_dense')(input_layer)
        x = Embedding(input_dim=1000, output_dim=self.embedding_dim, name='embedding_2')(x)
        
        # LSTM layer
        x = LSTM(100, name='lstm_2')(x)
        
        # Dense layers with dropout
        x = Dropout(0.2, name='dropout_1')(x)
        x = Dense(128, activation='relu', name='dense_2')(x)
        x = Dropout(0.2, name='dropout_2')(x)
        x = Dense(64, activation='relu', name='dense_3')(x)
        x = Dropout(0.2, name='dropout_3')(x)
        
        # Output layer
        output = Dense(1, name='dense_4')(x)
        
        # Create and compile model
        model = Model(inputs=input_layer, outputs=output, name='deep_lstm_model')
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', self._accuracy_metric]
        )
        
        return model

class StackedLSTMModel(BaseModel):
    """Third model architecture with stacked LSTM layers."""
    def build(self):
        # Input layer
        input_layer = Input(shape=(self.input_dim,), name='embedding_3_input')
        
        # Embedding layer
        x = Dense(1000, activation='relu', name='embedding_3_dense')(input_layer)
        x = Embedding(input_dim=1000, output_dim=self.embedding_dim, name='embedding_3')(x)
        
        # Stacked LSTM layers
        x = LSTM(100, return_sequences=True, name='lstm_3')(x)
        x = LSTM(100, name='lstm_4')(x)
        
        # Output layer
        output = Dense(1, name='dense_5')(x)
        
        # Create and compile model
        model = Model(inputs=input_layer, outputs=output, name='stacked_lstm_model')
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', self._accuracy_metric]
        )
        
        return model

class EnsembleModel(BaseModel):
    """Ensemble model combining all three architectures."""
    def build(self):
        # Create individual models
        simple_lstm = SimpleLSTMModel(self.input_dim, self.embedding_dim)
        deep_lstm = DeepLSTMModel(self.input_dim, self.embedding_dim)
        stacked_lstm = StackedLSTMModel(self.input_dim, self.embedding_dim)
        
        # Get their outputs
        output_1 = simple_lstm.build().output
        output_2 = deep_lstm.build().output
        output_3 = stacked_lstm.build().output
        
        # Combine outputs
        combined = concatenate([output_1, output_2, output_3])
        final_output = Dense(1, name='final_output')(combined)
        
        # Create and compile ensemble model
        model = Model(
            inputs=[
                simple_lstm.build().input,
                deep_lstm.build().input,
                stacked_lstm.build().input
            ],
            outputs=final_output,
            name='ensemble_model'
        )
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', self._accuracy_metric]
        )
        
        return model 