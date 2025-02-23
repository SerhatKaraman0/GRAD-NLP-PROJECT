from multiprocessing import get_context
import pandas as pd
import re
from typing import List
from nltk.tokenize import word_tokenize
import numpy as np
from langdetect import detect

# Use absolute imports
from common_imports import *  # Change from `..common_imports`
from .nlpmodel import NlpModel
from logging_config import *  # Change from `logging_config`
from utils.helper import CONTRACTIONS_DICT, SLANG_DICT  # Change from `utils.helper`


class PreprocessingModel(NlpModel):
    def __init__(self):
        super().__init__()
        self.SAVE_DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.df = self.cleaned_df.copy()  

        self.patterns = {
            'html': re.compile(r"<[^>]+>"),
            'url': re.compile(r"(http|ftp|https):\/\/[^\s/$.?#].[^\s]*"),
            'emoji': re.compile(r"[\U0001F600-\U0001F64F]"),
            'punctuation': re.compile(r"[!\'#\$%&\'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`{\|}~]"),
            'extra_punctuation': re.compile(r'(\W)\1+'),
            'whitespace': re.compile(r'\s+')
        }
        

        self.word_replacements = {**SLANG_DICT, **CONTRACTIONS_DICT}
        
    def _count_pattern_matches(self, pattern_name: str) -> tuple:
        self.logger.info("GETTING PATTERN MATCHES..")
        pattern = self.patterns[pattern_name]
        mask = self.df["Text"].astype(str).str.contains(pattern, regex=True, na=False)
        count = mask.sum()
        percentage = 100 * count / len(self.df)
        return count, percentage

    def get_statistics(self) -> None:
        """Efficiently calculate all statistics at once"""
        self.logger.info("GETTING STATISTICS..")
        for pattern_name in ['html', 'url', 'emoji']:
            count, percentage = self._count_pattern_matches(pattern_name)
            self.logger.info(f"Number of data containing {pattern_name}: {count} | Percentage: {percentage:.2f}%")

    @staticmethod
    def _process_chunk(args):
        """Process a chunk of texts in parallel"""
        texts, patterns, word_replacements = args
        processed = []
        tokenized = []
        
        for text in texts:
            text = str(text).lower()
            
            text = patterns['whitespace'].sub(' ', text)
            text = patterns['extra_punctuation'].sub(r'\1', text)
            text = patterns['punctuation'].sub('', text)
            text = patterns['html'].sub('', text)
            text = patterns['url'].sub('', text)
            
            words = text.split()
            words = [word_replacements.get(word, word) for word in words]
            processed_text = ' '.join(words)
            
            processed.append(processed_text)
            tokenized.append(word_tokenize(processed_text))
            
        return processed, tokenized

    def remove_foreign_words(self) -> None:
        """Remove non-English text efficiently"""
        self.logger.info("REMOVING FOREIGN WORDS STARTED..") 

        def detect_language(text):
            try:
                return detect(str(text)) == 'en'
            except:
                return False

        chunk_size = 1000
        mask = pd.Series(index=self.df.index, dtype=bool)
        
        for start in range(0, len(self.df), chunk_size):
            end = start + chunk_size
            chunk = self.df['Text'].iloc[start:end]
            mask.iloc[start:end] = chunk.apply(detect_language)
            
        self.df.loc[~mask, 'Text'] = ''

    def preprocess_dataframe(self) -> None:
        """Main preprocessing pipeline with M1-compatible parallel processing"""
        self.logger.info("PREPROCESSING STARTED..")

        chunk_size = 1000
        n_cores = os.cpu_count() or 4
        
        chunks = np.array_split(self.df['Text'], len(self.df) // chunk_size + 1)
        
        process_args = [(chunk, self.patterns, self.word_replacements) for chunk in chunks]
        
        with get_context('spawn').Pool(processes=n_cores) as pool:
            results = pool.map(self._process_chunk, process_args)
            
            processed_chunks, tokenized_chunks = zip(*results)
            
            processed_texts = [text for chunk in processed_chunks for text in chunk]
            tokenized_texts = [tokens for chunk in tokenized_chunks for tokens in chunk]

            self.df['Text'] = processed_texts
            self.df['Tokens'] = tokenized_texts
            
            self.df = self.df[['Id', 'Score', 'Summary', 'Text', 'Tokens']]

    def save_to_csv(self, output_path: str = "processed_data.csv") -> None:
        """Save the processed DataFrame to CSV"""
        self.logger.info("SAVING TO CSV STARTED..")

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        print(f"Saved CSV file to: {output_path}")

if __name__ == "__main__":
    # Ensure this code runs in a proper entry point for M1 compatibility
    model = PreprocessingModel()
    
    OUTPUT_DIR = os.path.join(model.SAVE_DATA_DIR, f"OUTPUT_Reviews_{datetime.now()}.csv")

    # model.preprocess_dataframe()

    # model.save_to_csv(output_path=OUTPUT_DIR)

    print(model.df.head())
    print("\nDataFrame columns:", model.df.columns.tolist())