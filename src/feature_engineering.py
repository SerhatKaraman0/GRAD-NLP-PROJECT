from src.common_imports import *  # noqa: F403, F405
from src.nlpmodel import NlpModel
from src.logging_config import *  # noqa: F403, F405
from utils.helper import CONTRACTIONS_DICT, SLANG_DICT  

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import swifter
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import pandas as pd
import gc
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split

# To ensure compatibility across environments
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up logging for better debugging and traceability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineering(NlpModel):
    def __init__(self):
        """Initialize paths and load dataset."""
        super().__init__()
        self.SAVE_DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.PREPROCESSED_DATA_DIR = os.path.join(self.SAVE_DATA_DIR, "PREPROCESSED_Reviews.csv")
        self.df = pd.read_csv(self.PREPROCESSED_DATA_DIR)
        self.df_size = len(self.df)
        self.batch_size = 10_000
        self.n_batches = (self.df_size + self.batch_size - 1) // self.batch_size
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=7000)  # Limit the vocabulary

    def preprocess_text(self):
        """Preprocess text data by handling NaNs and text cleaning."""
        self.df['Text'] = self.df['Text'].fillna('')  # Fill missing values with empty string

    def _vectorize_tfidf_batch(self, start_idx, end_idx, fit=False):
        """Vectorize a batch of data using TF-IDF."""
        batch = self.df.iloc[start_idx:end_idx]['Text']
        if fit:
            # Fit the vectorizer to the first batch
            self.vectorizer.fit(batch)
            fit = False  # After the first fit, don't need to fit again
        batch_tfidf = self.vectorizer.transform(batch)
        return batch_tfidf, fit
    
    def process_in_batches(self):
        """Process the data in batches and accumulate the sparse matrices."""
        self.preprocess_text()
        sparse_matrices = []
        fit = True  # Flag to fit only the first batch

        for start_idx in tqdm(range(0, self.df_size, self.batch_size), desc="Processing batches"):
            end_idx = min(start_idx + self.batch_size, self.df_size)
            batch_tfidf, fit = self._vectorize_tfidf_batch(start_idx, end_idx, fit)

            sparse_matrices.append(batch_tfidf)
            gc.collect()  # Clean up memory

        # Combine all sparse matrices into one sparse matrix
        all_tfidf_matrix = sparse.vstack(sparse_matrices)
        return all_tfidf_matrix
    
    def _vectorize_tfidf(self):
        """Vectorize the entire dataset."""
        return self.process_in_batches()

    def train_model(self, X, y):
        """Train a simple model on the TF-IDF matrix."""
        from tensorflow import keras
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Input
        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

        # Model architecture (Dense Feedforward Network)
        model = Sequential([
            Input(shape=(X.shape[1],)),  # Specify input shape using Input layer
            Dense(512, activation='relu'),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dense(1, activation='linear')  # Regression for rating prediction
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Set up callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ModelCheckpoint("sentiment_model_best.h5", monitor='val_loss', save_best_only=True)
        ]

        # Train the model
        model.fit(X, y, epochs=10, batch_size=512, validation_split=0.2, callbacks=callbacks)
        logger.info("Model Training Completed")

        return model

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the model on the test set."""
        loss, mae = model.evaluate(X_test, y_test)
        logger.info(f"Model Evaluation: Loss={loss}, MAE={mae}")
        return loss, mae

    def save_to_parquet(self, df, output_path: str = "processed_data.parquet") -> None:
        """Save the processed DataFrame to Parquet with gzip compression."""
        logger.info("Saving DataFrame to Parquet")

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to Parquet format with gzip compression
        df.to_parquet(f"{output_path}.parquet.gz", compression="gzip")

        logger.info(f"Data saved to {output_path}.parquet.gz")

    def load_and_process_data(self):
        """Load and process data, returning the TF-IDF matrix."""
        X_sparse = self._vectorize_tfidf()
        y = self.df["Score"].values  # Assuming 'Rating' is the target column
        
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_sparse, y, test_size=0.2, random_state=42)

        # Convert sparse matrix to dense (if needed for model training)
        X_train_dense = X_train.toarray()
        X_test_dense = X_test.toarray()
        
        return X_train_dense, y_train, X_test_dense, y_test


if __name__ == "__main__":
    # Initialize FeatureEngineering class
    model = FeatureEngineering()

    # Load data and process TF-IDF
    X_train, y_train, X_test, y_test = model.load_and_process_data()

    # Train the model
    trained_model = model.train_model(X_train, y_train)

    # Evaluate the model
    model.evaluate_model(trained_model, X_test, y_test)

    # Save the model (optional)
    trained_model.save("sentiment_model_final.h5")

    # Save TF-IDF scores to CSV (optional)
    tfidf_df = pd.DataFrame(trained_model.layers[0].get_weights()[0], columns=model.vectorizer.get_feature_names_out())
    tfidf_output_path = os.path.join(model.SAVE_DATA_DIR, "tfidf_scores.csv")
    tfidf_df.to_csv(tfidf_output_path, index=False)
    logger.info(f"TF-IDF scores saved to {tfidf_output_path}")
