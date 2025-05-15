from src.core.common_imports import * # noqa: F403, F405
from src.core.nlpmodel import NlpModel
from src.core.logging_config import *  # noqa: F403, F405
from utils.helper import CONTRACTIONS_DICT, SLANG_DICT

import pandas as pd
import numpy as np
import os
import sys
import gc
from tqdm import tqdm
import swifter
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse

class DataProcessor(NlpModel):
    """
    Class for loading, processing, and preparing data for NLP analysis
    """
    def __init__(self, batch_size=10000, max_features=10000):
        """
        Initialize the DataProcessor with configuration for data handling
        
        Args:
            batch_size (int): Size of batch for processing
            max_features (int): Maximum number of features for vectorization
        """
        super().__init__()
        self.SAVE_DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.STATS_DIR = os.path.join(self.BASE_DIR, "stats")
        self.PREPROCESSED_DATA_DIR = os.path.join(self.SAVE_DATA_DIR, "PREPROCESSED_Reviews.csv")
        
        self.df = pd.read_csv(self.PREPROCESSED_DATA_DIR)
        self.df_size = len(self.df)

        self.batch_size = batch_size
        self.n_batches = (self.df_size + self.batch_size - 1) // self.batch_size
        self.max_features = max_features

        self.vectorizer = CountVectorizer()
        
    def word_freq(self):
        """Calculate and visualize word frequency from the text data"""
        X = self.vectorizer.fit_transform(self.df['Text'])
        
        word_counts = X.sum(axis=0).A1
        feature_names = self.vectorizer.get_feature_names_out() 
        
        freq = dict(zip(feature_names, word_counts))
        
        sorted_freq = sorted(freq.items(), key=lambda item: item[1], reverse=True)

        # Convert to DataFrame for plotting
        freq_df = pd.DataFrame(sorted_freq[:10], columns=['Word', 'Frequency'])

        sns.barplot(x='Word', y='Frequency', data=freq_df)
        plt.xticks(rotation=45)
        plt.show()
        
    def create_bow(self):
        """For creating the bag of words from preprocessed data and saving the result into a DataFrame"""
        self.logger.info("BAG OF WORDS ACTION HAS STARTED..")

        # Ensure NaN values are handled
        self.df['Text'] = self.df['Text'].fillna('')
        
        # Initialize CountVectorizer with more aggressive feature reduction
        self.vectorizer = CountVectorizer(
            min_df=5,            # Ignore terms that appear in less than 5 documents
            max_df=0.5,          # Ignore terms that appear in more than 50% of documents
            max_features=self.max_features
        )
        
        self.vectorizer.fit(self.df['Text'])
        
        # Process in batches to reduce memory usage
        sparse_matrices = []
        
        for i in tqdm(range(self.n_batches), desc="Creating BoW"):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.df_size)
            
            batch = self.df.iloc[start_idx:end_idx]
            batch_sparse = self.vectorizer.transform(batch['Text'])
            sparse_matrices.append(batch_sparse)
            
            # Clear memory
            del batch
            gc.collect()
        
        sparse_matrices = sparse.vstack(sparse_matrices)
        feature_names = self.vectorizer.get_feature_names_out()
        bow_df = pd.DataFrame.sparse.from_spmatrix(sparse_matrices, columns=feature_names)

        self.bow_df = bow_df
        return bow_df
        
    def load_and_process_data(self):
        """Load and process the data including embedding preparation."""
        self.logger.info("Loading and processing data...")
        
        # Load preprocessed data
        df = pd.read_csv(self.PREPROCESSED_DATA_DIR)
        
        # Convert text to string and handle NaN values
        df['Text'] = df['Text'].fillna('').astype(str)
        
        # Use the original 1-5 score directly
        df['sentiment'] = df['Score']  # Use the actual 1-5 rating
        
        # Remove any rows with empty text
        df = df[df['Text'].str.strip() != '']
        
        # Use the 'Text' column for our analysis
        texts = df['Text'].values
        labels = df['sentiment'].values
        
        self.logger.info(f"Total samples: {len(texts)}")
        self.logger.info(f"Sample text: {texts[0][:100]}...")
        
        return texts, labels
    
    def save_to_parquet(self, df, output_path: str = "processed_data.parquet") -> None:
        """Save the processed DataFrame to Parquet with gzip compression"""
        self.logger.info("SAVING TO PARQUET STARTED..")
        self.print_section("SAVING TO PARQUET STARTED..")

        # Check if DataFrame has sparse data
        has_sparse = hasattr(df, 'sparse') and hasattr(df.sparse, 'to_dense')
    
        if has_sparse:
            # Convert sparse DataFrame to dense
            dense_df = df.sparse.to_dense()
        else:
            self.logger.warning("Input DataFrame does not contain sparse data.")
            dense_df = df
    
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
        # Save to Parquet format with gzip compression
        dense_df.to_parquet(f"{output_path}.parquet.gz", compression="gzip")

        self.logger.info(f"DF SAVED TO {output_path}.parquet.gz")

        return output_path
