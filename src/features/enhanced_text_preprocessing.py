#!/usr/bin/env python
# filepath: /Users/user/Desktop/Projects/NLP-Learning/src/features/enhanced_text_preprocessing.py

"""
Enhanced text preprocessing script with advanced NLP techniques.
This improves the quality of text representations for better model performance.
"""

import os
import sys
import re
import string
import unicodedata
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Import project modules
from src.core.logging_config import setup_logging, get_logger

# Setup logger
setup_logging()
logger = get_logger('Enhanced_Text_Processing')

class EnhancedTextProcessor:
    """Class for enhanced text preprocessing"""
    
    def __init__(self, remove_stopwords=False, lemmatize=True, keep_special_tokens=True):
        """
        Initialize the text processor
        
        Args:
            remove_stopwords (bool): Whether to remove stopwords
            lemmatize (bool): Whether to apply lemmatization
            keep_special_tokens (bool): Whether to keep special tokens like [URL], [EMAIL], etc.
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.keep_special_tokens = keep_special_tokens
        
        # Download required NLTK resources
        nltk_resources = ['punkt', 'stopwords', 'wordnet']
        for resource in nltk_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                logger.info(f"Downloading NLTK resource: {resource}")
                nltk.download(resource)
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Regex patterns for common text elements
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.phone_pattern = re.compile(r'\b(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b')
        self.number_pattern = re.compile(r'\b\d+\b')
        self.special_chars_pattern = re.compile(r'[^\w\s]')
        self.multiple_spaces_pattern = re.compile(r'\s+')
        
        logger.info("Initialized EnhancedTextProcessor")
    
    def normalize_unicode(self, text):
        """Normalize Unicode characters"""
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    def replace_urls(self, text):
        """Replace URLs with [URL] token"""
        return self.url_pattern.sub('[URL]', text) if self.keep_special_tokens else self.url_pattern.sub('', text)
    
    def replace_emails(self, text):
        """Replace emails with [EMAIL] token"""
        return self.email_pattern.sub('[EMAIL]', text) if self.keep_special_tokens else self.email_pattern.sub('', text)
    
    def replace_phone_numbers(self, text):
        """Replace phone numbers with [PHONE] token"""
        return self.phone_pattern.sub('[PHONE]', text) if self.keep_special_tokens else self.phone_pattern.sub('', text)
    
    def replace_numbers(self, text):
        """Replace numbers with [NUM] token"""
        return self.number_pattern.sub('[NUM]', text) if self.keep_special_tokens else self.number_pattern.sub('', text)
    
    def clean_text(self, text):
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Normalize unicode characters
        text = self.normalize_unicode(text)
        
        # Replace URLs, emails, and phone numbers
        text = self.replace_urls(text)
        text = self.replace_emails(text)
        text = self.replace_phone_numbers(text)
        text = self.replace_numbers(text)
        
        # Optionally remove special characters
        if not self.keep_special_tokens:
            text = self.special_chars_pattern.sub(' ', text)
        
        # Remove extra whitespace
        text = self.multiple_spaces_pattern.sub(' ', text)
        
        return text.strip()
    
    def advanced_process(self, text):
        """Apply advanced NLP processing"""
        # Skip if text is empty or NaN
        if not text or pd.isna(text):
            return ""
        
        # Basic cleaning
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if specified
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply lemmatization if specified
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Rejoin tokens
        return ' '.join(tokens)
    
    def process_dataframe(self, df, text_column, output_column=None):
        """
        Process text data in a DataFrame
        
        Args:
            df (pandas.DataFrame): Input DataFrame
            text_column (str): Column containing text data
            output_column (str): Column to store processed text (if None, overwrite input column)
            
        Returns:
            pandas.DataFrame: DataFrame with processed text
        """
        if output_column is None:
            output_column = text_column
        
        logger.info(f"Processing {len(df)} texts from column '{text_column}'")
        
        # Apply processing with progress bar
        df[output_column] = [
            self.advanced_process(text) 
            for text in tqdm(df[text_column], desc="Processing texts")
        ]
        
        logger.info(f"Text processing completed. Results saved in column '{output_column}'")
        return df

def main():
    """Main function to run text preprocessing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced text preprocessing")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file path")
    parser.add_argument("--text-column", type=str, default="Text", help="Column containing text data")
    parser.add_argument("--output-column", type=str, default="ProcessedText", help="Column to store processed text")
    parser.add_argument("--remove-stopwords", action="store_true", help="Remove stopwords")
    parser.add_argument("--no-lemmatize", action="store_true", help="Skip lemmatization")
    parser.add_argument("--no-special-tokens", action="store_true", help="Don't keep special tokens")
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    
    # Create processor
    processor = EnhancedTextProcessor(
        remove_stopwords=args.remove_stopwords,
        lemmatize=not args.no_lemmatize,
        keep_special_tokens=not args.no_special_tokens
    )
    
    # Process data
    processed_df = processor.process_dataframe(df, args.text_column, args.output_column)
    
    # Save processed data
    logger.info(f"Saving processed data to {args.output}")
    processed_df.to_csv(args.output, index=False)
    logger.info("Processing completed")

if __name__ == "__main__":
    main()
