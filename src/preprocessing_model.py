from multiprocessing import get_context
import pandas as pd
import re
from typing import List
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from langdetect import detect

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.common_imports import * 
from src.nlpmodel import NlpModel
import timeit
from src.logging_config import *  
from utils.helper import CONTRACTIONS_DICT, SLANG_DICT  


from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()


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
        table = Table(title="Pattern Statistics", show_header=True, header_style="bold magenta")
        table.add_column("Pattern", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Percentage", style="yellow")

        for pattern_name in ['html', 'url', 'emoji']:
            count, percentage = self._count_pattern_matches(pattern_name)
            table.add_row(pattern_name, str(count), f"{percentage:.2f}%")

        console.print(table)

    @staticmethod
    def _process_chunk(args):
        """Process a chunk of texts in parallel using numpy for loop"""
        texts, patterns, word_replacements = args

        if len(texts) == 0:
            return [], []  # Return empty lists for empty chunks

        processed = np.empty(len(texts), dtype=object)
        tokenized = np.empty(len(texts), dtype=object)
        
        for i in range(len(texts)):
            text = str(texts.iloc[i]).lower()  # Use .iloc to access by position
            
            text = patterns['whitespace'].sub(' ', text)
            text = patterns['extra_punctuation'].sub(r'\1', text)
            text = patterns['punctuation'].sub('', text)
            text = patterns['html'].sub('', text)
            text = patterns['url'].sub('', text)
            
            words = text.split()
            words = [word_replacements.get(word, word) for word in words]
            processed_text = ' '.join(words)
            
            processed[i] = processed_text
            tokenized[i] = word_tokenize(processed_text)
            
        return processed.tolist(), tokenized.tolist()

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

        if self.df.empty or self.df['Text'].empty:
            self.logger.error("DataFrame or 'Text' column is empty. Aborting preprocessing.")
            return

        chunk_size = 1000
        n_cores = os.cpu_count() or 4
        
        chunks = np.array_split(self.df['Text'], len(self.df) // chunk_size + 1)
        
        process_args = [(chunk, self.patterns, self.word_replacements) for chunk in chunks]
        
        with get_context('spawn').Pool(processes=n_cores) as pool:
            results = list(track(
                pool.imap(self._process_chunk, process_args),
                total=len(process_args),
                description="Processing chunks..."
            ))
            
            stop_words = set(stopwords.words('english'))
            processed_chunks, tokenized_chunks = zip(*results)
            
            processed_texts = [text for chunk in processed_chunks for text in chunk]
            
            # Tokenization and removing stop words 
            tokenized_texts = [tokens for chunk in tokenized_chunks for tokens in chunk if tokens not in stop_words]

            self.df['Text'] = processed_texts
            self.df['Tokens'] = tokenized_texts
            
            self.df = self.df[['Id', 'Score', 'Summary', 'Text', 'Tokens']]

    def save_to_csv(self, output_path: str = "processed_data.csv") -> None:
        """Save the processed DataFrame to CSV"""
        self.logger.info("SAVING TO CSV STARTED..")

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        console.print(f"[bold green]Saved CSV file to: {output_path}[/bold green]")


if __name__ == "__main__":
    model = PreprocessingModel()
    
    console.print("[bold cyan]DataFrame shape:[/bold cyan]", model.df.shape)
    console.print("[bold cyan]First few rows of 'Text' column:[/bold cyan]")
    console.print(model.df['Text'].head())
    
    OUTPUT_DIR = os.path.join(model.SAVE_DATA_DIR, "PREPROCESSED_Reviews.csv")

    start_time = timeit.default_timer()
    model.preprocess_dataframe()
    elapsed = timeit.default_timer() - start_time

    model.save_to_csv(output_path=OUTPUT_DIR)

    console.print(f"[bold green]Preprocessing took {elapsed:.2f} seconds[/bold green]")
    console.print("[bold cyan]Processed DataFrame:[/bold cyan]")
    console.print(model.df.head())
    console.print("\n[bold cyan]DataFrame columns:[/bold cyan]", model.df.columns.tolist())