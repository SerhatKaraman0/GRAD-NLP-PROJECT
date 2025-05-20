from src.core.common_imports import * # noqa: F403, F405
from src.core.nlpmodel import NlpModel
from src.core.logging_config import *  # noqa: F403, F405
from utils.helper import CONTRACTIONS_DICT, SLANG_DICT
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords, wordnet
import nltk
import gc
import itertools
from multiprocessing import get_context
import swifter
from tqdm import tqdm
import string 
from textblob import TextBlob
from bs4 import BeautifulSoup
import spacy
from langdetect import detect
import emoji
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import cProfile
import pstats
import timeit
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import track
from functools import partial
import re
import os
import sys

# Download necessary NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

# Load spaCy model for advanced NLP tasks
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

warnings.filterwarnings("ignore", category=FutureWarning)

console = Console()


def get_wordnet_pos(word, tag):
    """Map POS tag to wordnet POS tag for proper lemmatization"""
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }
    return tag_dict.get(tag[0].upper(), wordnet.NOUN)


class PreprocessingModel(NlpModel):
    __slots__ = [
        "SAVE_DATA_DIR", "STATS_DIR", "df", "patterns", 
        "word_replacements", "lemmatizer", "stemmer", "tweet_tokenizer",
        "custom_stop_words", "use_lemmatization", "use_stemming", 
        "preserve_negation", "preserve_named_entities", "preserve_numbers",
        "correct_spelling", "advanced_tokenization", "use_spacy"
    ]

    def __init__(self, 
                 use_lemmatization=True, 
                 use_stemming=False,
                 preserve_negation=True,
                 preserve_named_entities=True,
                 preserve_numbers=True,
                 correct_spelling=True,
                 advanced_tokenization=True,
                 use_spacy=True):
        """
        Initialize the preprocessing model with configurable options
        
        Args:
            use_lemmatization (bool): Whether to use lemmatization
            use_stemming (bool): Whether to use stemming (not recommended with lemmatization)
            preserve_negation (bool): Whether to preserve negation words
            preserve_named_entities (bool): Whether to preserve named entities
            preserve_numbers (bool): Whether to preserve numeric information
            correct_spelling (bool): Whether to apply spelling correction
            advanced_tokenization (bool): Whether to use advanced tokenization
            use_spacy (bool): Whether to use spaCy for advanced NLP tasks
        """
        super().__init__()
        self.SAVE_DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.STATS_DIR = os.path.join(self.BASE_DIR, "stats")
        self.df = self.cleaned_df.copy() if hasattr(self, 'cleaned_df') else None

        # Initialize preprocessing options
        self.use_lemmatization = use_lemmatization
        self.use_stemming = use_stemming
        self.preserve_negation = preserve_negation
        self.preserve_named_entities = preserve_named_entities
        self.preserve_numbers = preserve_numbers
        self.correct_spelling = correct_spelling
        self.advanced_tokenization = advanced_tokenization
        self.use_spacy = use_spacy
        
        # Initialize NLP tools
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        self.stemmer = PorterStemmer() if use_stemming else None
        self.tweet_tokenizer = TweetTokenizer() if advanced_tokenization else None
        
        # Consolidated and optimized regex patterns
        self.patterns = {
            'url': re.compile(r"(https?|ftp):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])"),
            'html': re.compile(r'<.*?>'),
            'extra_whitespace': re.compile(r'\s+'),
            'special_characters': re.compile(r"[^\w\s]"),
            'numbers': re.compile(r"\b\d+\b"),
            'special_chars': re.compile(r"[^\w\s]"),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'user_mentions': re.compile(r'@\w+'),
            'hashtags': re.compile(r'#\w+')
        }
        
        # Enhance word replacements dictionary
        self.word_replacements = {**SLANG_DICT, **CONTRACTIONS_DICT}
        
        # Define custom stop words
        self.custom_stop_words = set(stopwords.words('english'))
        
        # Remove negation words from stopwords if needed
        if self.preserve_negation:
            negation_words = {'no', 'not', 'nor', 'neither', 'never', 'none'}
            self.custom_stop_words = self.custom_stop_words - negation_words
    
    def _count_pattern_matches(self, pattern_name: str, df) -> tuple:
        self.logger.info(f"GETTING {pattern_name.upper()} PATTERN MATCHES..")
        
        pattern = self.patterns[pattern_name]
        mask = df["Text"].astype(str).str.contains(pattern, regex=True, na=False)
        count = mask.sum()
        percentage = 100 * count / len(df)
        return count, percentage
    
    def get_statistics(self, df) -> None:
        """Efficiently calculate all statistics at once"""
        
        self.logger.info("GETTING STATISTICS..")
        self.print_section("GETTING STATISTICS..")

        table = Table(title="Pattern Statistics", show_header=True, header_style="bold magenta")
        table.add_column("Pattern", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Percentage", style="yellow")

        for pattern_name in self.patterns.keys():
            count, percentage = self._count_pattern_matches(pattern_name, df)
            table.add_row(pattern_name, str(count), f"{percentage:.2f}%")

        console.print(table)
        
        # Add more advanced statistics
        if 'Tokens' in df.columns and 'Sentences' in df.columns:
            stats_table = Table(title="Text Statistics", show_header=True, header_style="bold blue")
            stats_table.add_column("Statistic", style="cyan")
            stats_table.add_column("Value", style="green")
            
            avg_tokens = df['Tokens'].apply(len).mean()
            avg_sentences = df['Sentences'].apply(len).mean()
            avg_token_length = df['Tokens'].apply(lambda x: np.mean([len(t) for t in x]) if x else 0).mean()
            
            stats_table.add_row("Average Tokens per Text", f"{avg_tokens:.2f}")
            stats_table.add_row("Average Sentences per Text", f"{avg_sentences:.2f}")
            stats_table.add_row("Average Token Length", f"{avg_token_length:.2f}")
            
            console.print(stats_table)

    def _process_text_with_spacy(self, text):
        """Process text using spaCy for advanced NLP tasks"""
        doc = nlp(text)
        
        # Extract named entities if needed
        entities = {ent.text: ent.label_ for ent in doc.ents} if self.preserve_named_entities else {}
        
        # Get tokens with lemmatization
        tokens = []
        for token in doc:
            # Skip stop words but preserve special cases
            if token.is_stop and token.text.lower() not in self.custom_stop_words:
                continue
                
            # Handle numbers
            if token.like_num and not self.preserve_numbers:
                continue
                
            # Include lemma or original token
            tokens.append(token.lemma_ if self.use_lemmatization else token.text)
        
        # Get sentences
        sentences = [sent.text for sent in doc.sents]
        
        # Reconstruct processed text
        processed_text = " ".join(tokens)
        
        return processed_text, tokens, sentences, entities

    def _process_text(self, text):
        """Process a single text - for use with swifter"""
        
        if not text or pd.isna(text):
            return "", [], [], {}
        
        text = str(text).lower()
        
        # Convert emojis to text
        text = emoji.demojize(text)
        
        # Clean HTML content more effectively
        text = BeautifulSoup(text, "html.parser", features="lxml").get_text()
        
        # Remove URLs, emails, user mentions
        text = self.patterns['url'].sub(' ', text)
        text = self.patterns['email'].sub(' ', text)
        text = self.patterns['user_mentions'].sub(' ', text)
        
        # Handle numbers based on configuration
        if not self.preserve_numbers:
            text = self.patterns['numbers'].sub(' ', text)
        
        # Remove special characters but handle hashtags differently
        hashtags = re.findall(self.patterns['hashtags'], text)
        text = self.patterns['special_chars'].sub(' ', text)
        
        # Add back hashtags without the # symbol if they represent meaningful content
        for tag in hashtags:
            text += f" {tag[1:]} "
        
        # Normalize whitespace
        text = " ".join(text.split())
        
        # Use SpaCy for advanced NLP processing if enabled
        if self.use_spacy:
            return self._process_text_with_spacy(text)
        
        # Otherwise use traditional NLTK-based approach
        # Replace contractions and slang
        words = text.split()
        words = [self.word_replacements.get(word, word) for word in words]
        text = ' '.join(words)
        
        # Tokenize using basic or advanced tokenizer
        if self.advanced_tokenization:
            tokens = self.tweet_tokenizer.tokenize(text)
        else:
            tokens = word_tokenize(text)
        
        # Apply POS tagging for better lemmatization if needed
        if self.use_lemmatization:
            pos_tags = nltk.pos_tag(tokens)
            tokens = [self.lemmatizer.lemmatize(word, get_wordnet_pos(word, tag)) 
                    for word, tag in pos_tags]
        
        # Apply stemming if needed (not recommended with lemmatization)
        elif self.use_stemming:
            tokens = [self.stemmer.stem(word) for word in tokens]
        
        # Remove stopwords while preserving negation words if configured
        tokens = [token for token in tokens 
                if token not in self.custom_stop_words]
        
        # Get sentences
        sentences = sent_tokenize(' '.join(words))
        
        # Reconstruct processed text
        processed_text = ' '.join(tokens)
        
        # Correct spelling if enabled
        if self.correct_spelling:
            processed_text = str(TextBlob(processed_text).correct())
            # Re-tokenize after spelling correction
            if self.advanced_tokenization:
                tokens = self.tweet_tokenizer.tokenize(processed_text)
            else:
                tokens = word_tokenize(processed_text)
        
        # No entities in non-spaCy mode
        entities = {}
        
        return processed_text, tokens, sentences, entities

    @staticmethod
    def _process_chunk(args):
        """Process a chunk of texts in parallel"""
        
        texts, preprocessor = args
        if len(texts) == 0:
            return [], [], [], []
        
        if isinstance(texts, list):
            texts = pd.Series(texts)
        
        processed = np.empty(len(texts), dtype=object)
        tokenized = np.empty(len(texts), dtype=object)
        sentence_tokenized = np.empty(len(texts), dtype=object)
        entities = np.empty(len(texts), dtype=object)
        
        for i in range(len(texts)):
            text = str(texts.iloc[i]).lower()
            
            # Process text using the preprocessor's method
            processed_text, tokens, sentences, ents = preprocessor._process_text(text)
            
            processed[i] = processed_text
            tokenized[i] = tokens
            sentence_tokenized[i] = sentences
            entities[i] = ents
            
        return processed.tolist(), tokenized.tolist(), sentence_tokenized.tolist(), entities.tolist()
   
    @staticmethod
    def _replace_words(text, word_replacements):
        """Replace words using the replacement dictionary"""
        words = text.split()
        words = [word_replacements.get(word, word) for word in words]
        return ' '.join(words)
   
    @staticmethod
    def process_with_timeout(args):
        """Wrapper function with timeout handling"""
        try:
            return PreprocessingModel._process_chunk(args)
        except Exception as e:
            print(f"Error in worker process: {e}")
            return [], [], [], []
        
    def preprocess_dataframe(self) -> None:
        """Main preprocessing pipeline with improved parallelization"""
        self.logger.info("PREPROCESSING STARTED..")
        self.print_section("PREPROCESSING STARTED..")
        
        if self.df is None or self.df.empty or 'Text' not in self.df.columns:
            self.logger.error("DataFrame or 'Text' column is empty. Aborting preprocessing.")
            return
        
        # Determine processing approach based on data size
        if len(self.df) > 10000:  # For large datasets, use chunked approach
            total_rows = len(self.df)
            rows_per_chunk = 1000  # Adjust based on memory capacity
            n_chunks = max(1, int(total_rows / rows_per_chunk))
            
            n_cores = os.cpu_count() or 4
            n_cores = min(n_cores, 8)
            
            chunks = np.array_split(self.df['Text'], n_chunks)
            process_args = [(chunk, self) for chunk in chunks]
            
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                # Create futures with map
                futures = []
                for args in process_args:
                    futures.append(executor.submit(PreprocessingModel.process_with_timeout, args))
                
                # Process results as they complete
                processed_texts = []
                tokenized_texts = []
                sentence_tokenized_texts = []
                entities_list = []
                
                for future in track(futures, description="Processing chunks..."):
                    try:
                        # Add a timeout to prevent hanging processes
                        result = future.result(timeout=300)  
                        if result and all(result):
                            proc_chunk, token_chunk, sent_chunk, ent_chunk = result
                            processed_texts.extend(proc_chunk)
                            tokenized_texts.extend(token_chunk)
                            sentence_tokenized_texts.extend(sent_chunk)
                            entities_list.extend(ent_chunk)
                    except TimeoutError:
                        self.logger.warning("A worker process timed out and will be skipped")
                        console.print("[bold yellow]A worker process timed out and will be skipped[/bold yellow]")
                    except Exception as e:
                        self.logger.error(f"Error processing chunk: {e}")
                        console.print(f"[bold red]Error processing chunk: {e}[/bold red]")
                
                # Force garbage collection
                gc.collect()
        else:
            # For smaller datasets, use swifter for parallelized apply
            console.print("[bold yellow]Using swifter for parallel processing...[/bold yellow]")
            
            # Process texts and unpack results
            results = self.df['Text'].swifter.progress_bar(True).apply(self._process_text)
            
            # Unpack results
            processed_texts, tokenized_texts, sentence_tokenized_texts, entities_list = zip(*results)
        
        # Create new DataFrame columns
        new_data = {
            'Text': processed_texts,
            'Tokens': tokenized_texts,
            'Sentences': sentence_tokenized_texts,
            'Entities': entities_list,
            'Token_count': [len(tokens) for tokens in tokenized_texts],
            'Sentence_count': [len(sentences) for sentences in sentence_tokenized_texts],
            'Text_length': [len(text) for text in processed_texts]
        }
        
        # Add sentiment analysis if TextBlob is available
        try:
            sentiment_scores = [TextBlob(text).sentiment.polarity for text in processed_texts]
            subjectivity_scores = [TextBlob(text).sentiment.subjectivity for text in processed_texts]
            new_data['Sentiment'] = sentiment_scores
            new_data['Subjectivity'] = subjectivity_scores
        except:
            self.logger.warning("TextBlob sentiment analysis failed, skipping sentiment features")
        
        # Preserve original columns and add new ones
        preserve_columns = ['Id', 'Score', 'Summary'] if all(col in self.df.columns for col in ['Id', 'Score', 'Summary']) else []
        self.df = self.df[preserve_columns].assign(**new_data) if preserve_columns else pd.DataFrame(new_data)
            
        # Final cleanup
        gc.collect()
        
        self.logger.info("Preprocessing completed successfully")
        console.print("[bold green]Preprocessing completed successfully[/bold green]")

    def remove_foreign_words(self) -> None:
        """Remove non-English text efficiently using langdetect"""
        self.logger.info("REMOVING FOREIGN WORDS STARTED..") 
        self.print_section("REMOVING FOREIGN WORDS STARTED..") 

        def detect_language(text):
            if not text or pd.isna(text) or len(str(text).strip()) < 10:
                return True  # Keep very short texts
            try:
                return detect(str(text)) == 'en'
            except:
                return True  # Keep texts where language detection fails
        
        # Using swifter for parallelized language detection
        is_english = self.df['Text'].swifter.apply(detect_language)
        non_english_count = (~is_english).sum()
        
        self.logger.info(f"Found {non_english_count} non-English texts out of {len(self.df)} ({non_english_count/len(self.df)*100:.2f}%)")
        console.print(f"[yellow]Found {non_english_count} non-English texts out of {len(self.df)} ({non_english_count/len(self.df)*100:.2f}%)[/yellow]")
        
        # Replace non-English text with empty string or remove
        self.df.loc[~is_english, 'Text'] = ''

    def save_to_csv(self, output_path: str = "processed_data.csv") -> None:
        """Save the processed DataFrame to CSV"""
        self.logger.info("SAVING TO CSV STARTED..")
        self.print_section("SAVING TO CSV STARTED..")

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        console.print(f"[bold green]Saved CSV file to: {output_path}[/bold green]")
        
    def save_to_parquet(self, output_path: str = "processed_data.parquet") -> None:
        """Save the processed DataFrame to Parquet with compression"""
        self.logger.info("SAVING TO PARQUET STARTED..")
        self.print_section("SAVING TO PARQUET STARTED..")
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        self.df.to_parquet(output_path, compression='gzip')
        console.print(f"[bold green]Saved Parquet file to: {output_path}[/bold green]")

    def perform_feature_engineering(self):
        """Perform basic feature engineering on preprocessed data"""
        if self.df is None or self.df.empty:
            self.logger.error("DataFrame is empty. Cannot perform feature engineering.")
            return
            
        self.logger.info("Performing feature engineering")
        console.print("[bold cyan]Performing feature engineering...[/bold cyan]")
        
        # Text length features
        self.df['Text_length'] = self.df['Text'].str.len()
        self.df['Word_count'] = self.df['Text'].str.split().str.len()
        
        # Average word length
        self.df['Avg_word_length'] = self.df['Text'].apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if x else 0
        )
        
        # Text complexity features (if TextBlob is available)
        try:
            # Lexical diversity (unique words / total words)
            self.df['Lexical_diversity'] = self.df['Tokens'].apply(
                lambda x: len(set(x)) / len(x) if x and len(x) > 0 else 0
            )
            
            # Named entity count
            if 'Entities' in self.df.columns:
                self.df['Entity_count'] = self.df['Entities'].apply(len)
        except:
            self.logger.warning("Some feature engineering steps failed")
        
        console.print("[bold green]Feature engineering completed[/bold green]")
        return self.df
    
    def get_preprocessing_summary(self):
        """Get a summary of preprocessing configuration and results"""
        summary = Table(title="Preprocessing Summary", show_header=True, header_style="bold green")
        summary.add_column("Setting", style="cyan")
        summary.add_column("Value", style="yellow")
        
        # Add preprocessing settings
        summary.add_row("Lemmatization", str(self.use_lemmatization))
        summary.add_row("Stemming", str(self.use_stemming))
        summary.add_row("Preserve Negation", str(self.preserve_negation))
        summary.add_row("Preserve Named Entities", str(self.preserve_named_entities))
        summary.add_row("Preserve Numbers", str(self.preserve_numbers))
        summary.add_row("Spelling Correction", str(self.correct_spelling))
        summary.add_row("Advanced Tokenization", str(self.advanced_tokenization))
        summary.add_row("Use spaCy", str(self.use_spacy))
        
        # Add processing results if available
        if self.df is not None and not self.df.empty:
            summary.add_row("Processed Rows", str(len(self.df)))
            if 'Token_count' in self.df.columns:
                summary.add_row("Avg Token Count", f"{self.df['Token_count'].mean():.2f}")
            if 'Sentence_count' in self.df.columns:
                summary.add_row("Avg Sentence Count", f"{self.df['Sentence_count'].mean():.2f}")
            if 'Sentiment' in self.df.columns:
                summary.add_row("Avg Sentiment", f"{self.df['Sentiment'].mean():.3f}")
        
        console.print(summary)


if __name__ == "__main__":
    with cProfile.Profile() as profile:
        # Create preprocessing model with enhanced options
        model = PreprocessingModel(
            use_lemmatization=True,
            use_stemming=False,
            preserve_negation=True, 
            preserve_named_entities=True,
            preserve_numbers=True,
            correct_spelling=True,
            advanced_tokenization=True,
            use_spacy=True  # Set to False for faster but less accurate processing
        )
        
        console.print("[bold cyan]DataFrame shape:[/bold cyan]", model.df.shape)
        console.print("[bold cyan]First few rows of 'Text' column:[/bold cyan]")
        console.print(model.df['Text'].head())
        
        model.get_statistics(model.df)

        # Output paths for different formats
        OUTPUT_CSV = os.path.join(model.SAVE_DATA_DIR, "PREPROCESSED_Reviews.csv")
        OUTPUT_PARQUET = os.path.join(model.SAVE_DATA_DIR, "PREPROCESSED_Reviews.parquet")

        # Preprocess data
        start_time = timeit.default_timer()
        model.preprocess_dataframe()
        elapsed = timeit.default_timer() - start_time

        # Save to multiple formats
        model.save_to_csv(output_path=OUTPUT_CSV)
        model.save_to_parquet(output_path=OUTPUT_PARQUET)

        # Perform feature engineering
        model.perform_feature_engineering()
        
        # Load processed data and get statistics
        cleaned_df = pd.read_csv(OUTPUT_CSV)
        model.get_statistics(cleaned_df)
        
        # Print processing summary
        model.get_preprocessing_summary()

        console.print(f"[bold green]Preprocessing took {elapsed:.2f} seconds[/bold green]")
        console.print("[bold cyan]Processed DataFrame:[/bold cyan]")
        console.print(model.df.head())
        console.print("\n[bold cyan]DataFrame columns:[/bold cyan]", model.df.columns.tolist())

    stats_file_dir = os.path.join(model.STATS_DIR, "results.prof")
    
    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)

    results.dump_stats(stats_file_dir)