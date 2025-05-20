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
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import matplotlib.gridspec as gridspec

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
        self.metrics_dir = os.path.join(self.SAVE_DATA_DIR, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        
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
    
    def create_word_frequency_by_rating(self):
        """
        Generate word frequency visualization by rating
        
        Returns:
            str: Path to the saved visualization
        """
        self.logger.info("Generating word frequency by rating visualization")
        
        # Ensure all texts are strings
        self.df['Text'] = self.df['Text'].fillna('').astype(str)
        
        plt.figure(figsize=(12, 10))
        
        # Top frequent words for each rating
        top_n = 10
        ratings = sorted(self.df['Score'].unique())
        
        # Create a subplot for each rating
        fig, axes = plt.subplots(len(ratings), 1, figsize=(12, 5 * len(ratings)))
        
        for i, rating in enumerate(ratings):
            # Filter by rating
            rating_df = self.df[self.df['Score'] == rating]
            
            # Vectorize the text for this rating
            vectorizer = CountVectorizer(stop_words='english', min_df=5, max_df=0.7)
            X = vectorizer.fit_transform(rating_df['Text'])
            
            # Get word frequencies
            word_counts = X.sum(axis=0).A1
            feature_names = vectorizer.get_feature_names_out()
            
            # Create frequency dictionary and sort
            freq = dict(zip(feature_names, word_counts))
            sorted_freq = sorted(freq.items(), key=lambda item: item[1], reverse=True)[:top_n]
            
            # Create DataFrame for plotting
            freq_df = pd.DataFrame(sorted_freq, columns=['Word', 'Frequency'])
            
            # Plot on the corresponding subplot
            sns.barplot(x='Frequency', y='Word', data=freq_df, ax=axes[i], palette='viridis')
            axes[i].set_title(f'{rating}-Star Reviews: Top {top_n} Words')
            axes[i].set_xlabel('Frequency')
            axes[i].set_ylabel('Word')
        
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(self.metrics_dir, 'word_frequency_by_rating.png')
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"Word frequency by rating visualization saved at {output_path}")
        return output_path
    
    def create_review_length_by_rating(self):
        """
        Generate visualization of review length by rating
        
        Returns:
            str: Path to the saved visualization
        """
        self.logger.info("Generating review length by rating visualization")
        
        # Calculate review length for each row
        self.df['review_length'] = self.df['Text'].fillna('').astype(str).apply(len)
        
        # Create a boxplot of review length by rating
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Score', y='review_length', data=self.df, palette='viridis')
        
        # Add a swarmplot to show individual points
        sns.swarmplot(x='Score', y='review_length', data=self.df.sample(min(1000, len(self.df))), 
                      color='black', alpha=0.5, size=3)
        
        plt.title('Review Length by Rating')
        plt.xlabel('Rating')
        plt.ylabel('Review Length (characters)')
        
        # Save the figure
        output_path = os.path.join(self.metrics_dir, 'review_length_by_rating.png')
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"Review length by rating visualization saved at {output_path}")
        return output_path
    
    # Helper function moved outside of method to support multiprocessing
    def _generate_wordcloud_for_rating(self, rating, df, max_samples):
        """Helper function to generate wordcloud for a single rating
        
        Args:
            rating: The rating value to generate wordcloud for
            df: DataFrame containing the data
            max_samples: Maximum number of samples to use
            
        Returns:
            tuple: (rating, wordcloud object)
        """
        self.logger.info(f"Generating word cloud for rating {rating}")
        
        # Filter by rating
        rating_df = df[df['Score'] == rating]
        
        # Sample if dataset is too large
        if len(rating_df) > max_samples:
            self.logger.info(f"Sampling {max_samples} reviews from {len(rating_df)} for rating {rating}")
            rating_df = rating_df.sample(max_samples, random_state=42)
        
        # Combine all text for this rating
        text = ' '.join(rating_df['Text'].fillna('').astype(str).values)
        
        # Create a word cloud with optimized parameters
        wordcloud = WordCloud(
            width=800, height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis',
            contour_width=1,
            prefer_horizontal=0.9,  # Allow more vertical words
            relative_scaling=0.5    # Scale word size by frequency^0.5 instead of frequency
        ).generate(text)
        
        return rating, wordcloud
    
    def create_wordclouds_by_rating(self, max_samples=5000, use_multiprocessing=True):
        """
        Generate word clouds for each rating
        
        Args:
            max_samples (int): Maximum number of reviews to sample per rating
            use_multiprocessing (bool): Whether to use multiprocessing for parallel generation
            
        Returns:
            str: Path to the saved visualization
        """
        self.logger.info("Generating word clouds by rating")
        import multiprocessing as mp
        from functools import partial
        
        # Create a figure with a subplot for each rating
        ratings = sorted(self.df['Score'].unique())
        
        fig = plt.figure(figsize=(15, 4 * len(ratings)))
        gs = gridspec.GridSpec(len(ratings), 1)
        
        # Use multiprocessing if enabled and we have multiple ratings
        if use_multiprocessing and len(ratings) > 1:
            self.logger.info("Using multiprocessing for wordcloud generation")
            try:
                # Create a wrapper function that doesn't use self directly
                def process_rating(rating, df=self.df, max_samples=max_samples):
                    # Filter by rating
                    rating_df = df[df['Score'] == rating]
                    
                    # Sample if dataset is too large
                    if len(rating_df) > max_samples:
                        rating_df = rating_df.sample(max_samples, random_state=42)
                    
                    # Combine all text for this rating
                    text = ' '.join(rating_df['Text'].fillna('').astype(str).values)
                    
                    # Create a word cloud with optimized parameters
                    wordcloud = WordCloud(
                        width=800, height=400, 
                        background_color='white',
                        max_words=100,
                        colormap='viridis',
                        contour_width=1,
                        prefer_horizontal=0.9,
                        relative_scaling=0.5
                    ).generate(text)
                    
                    return rating, wordcloud

                with mp.Pool(min(mp.cpu_count(), len(ratings))) as pool:
                    results = pool.map(process_rating, ratings)
            except Exception as e:
                self.logger.warning(f"Multiprocessing failed: {e}. Falling back to sequential processing.")
                results = [self._generate_wordcloud_for_rating(rating, self.df, max_samples) for rating in ratings]
                
            # Sort results by rating
            results = sorted(results, key=lambda x: x[0])
            
            # Plot each word cloud
            for i, (rating, wordcloud) in enumerate(results):
                ax = plt.subplot(gs[i])
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(f'Word Cloud for {rating}-Star Reviews')
                ax.axis('off')
        else:
            # Sequential processing
            for i, rating in enumerate(ratings):
                rating, wordcloud = self._generate_wordcloud_for_rating(rating, self.df, max_samples)
                
                # Plot the word cloud
                ax = plt.subplot(gs[i])
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(f'Word Cloud for {rating}-Star Reviews')
                ax.axis('off')
        
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(self.metrics_dir, 'wordclouds_by_rating.png')
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"Word clouds by rating visualization saved at {output_path}")
        return output_path
    
    def create_sentiment_distribution(self):
        """
        Generate visualization of sentiment distribution across ratings
        
        Returns:
            str: Path to the saved visualization
        """
        self.logger.info("Generating sentiment distribution visualization")
        
        # Count the number of reviews for each rating
        rating_counts = self.df['Score'].value_counts().sort_index()
        
        # Create a bar chart
        plt.figure(figsize=(10, 6))
        sns.barplot(x=rating_counts.index, y=rating_counts.values, palette='viridis')
        
        plt.title('Distribution of Reviews by Rating')
        plt.xlabel('Rating')
        plt.ylabel('Number of Reviews')
        
        # Add counts on top of each bar
        for i, count in enumerate(rating_counts.values):
            plt.text(i, count + 100, f'{count:,}', ha='center')
        
        # Save the figure
        output_path = os.path.join(self.metrics_dir, 'sentiment_distribution.png')
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"Sentiment distribution visualization saved at {output_path}")
        return output_path
    
    def generate_rating_visualizations(self):
        """
        Generate all rating-based visualizations
        
        Returns:
            dict: Paths to all generated visualizations
        """
        self.logger.info("Generating all rating visualizations")
        
        visualizations = {
            'word_frequency': self.create_word_frequency_by_rating(),
            'review_length': self.create_review_length_by_rating(),
            'wordclouds': self.create_wordclouds_by_rating(),
            'sentiment_distribution': self.create_sentiment_distribution()
        }
        
        return visualizations
        
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
