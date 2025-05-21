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
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
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
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
from dateutil import parser as date_parser
import unicodedata
import warnings

# Set seed for language detection to ensure consistent results
DetectorFactory.seed = 42

# Download necessary NLTK resources if not already downloaded
nltk_resources = [
    'punkt',
    'stopwords',
    'wordnet',
    'averaged_perceptron_tagger',
    'punkt_tab'
]

for resource in nltk_resources:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
    except LookupError:
        print(f"Downloading {resource}...")
        nltk.download(resource)

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


def normalize_text(text):
    """
    Perform advanced text normalization including unicode normalization
    and handling of special character representations
    """
    if not text or pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    # Fix common encoding issues
    text = text.replace('â€™', "'")
    text = text.replace('â€œ', '"')
    text = text.replace('â€', '"')
    text = text.replace('â€"', '—')
    text = text.replace('â€"', '-')
    
    return text


def normalize_numbers_and_dates(text):
    """
    Normalize numbers, dates, and measurements in text while preserving meaning
    """
    if not text or pd.isna(text):
        return ""
    
    # Normalize number formats (e.g., 1,000,000 -> 1000000)
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    
    # Normalize decimal formats (e.g., "0.5" and "0,5" -> "0.5")
    text = re.sub(r'(\d),(\d)', r'\1.\2', text)
    
    # Normalize date formats to ISO format where possible
    date_patterns = [
        # MM/DD/YYYY or DD/MM/YYYY
        r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b',
        # Month name, day, year
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (\d{1,2}),? (\d{4})\b'
    ]
    
    for pattern in date_patterns:
        dates = re.findall(pattern, text)
        for date_match in dates:
            try:
                # Try to parse the date
                date_str = ' '.join(date_match).strip()
                parsed_date = date_parser.parse(date_str)
                # Replace with a standardized format
                text = text.replace(date_str, parsed_date.strftime('%Y-%m-%d'))
            except:
                # If parsing fails, keep original
                pass
    
    # Normalize common measurements
    # E.g., "5 kg", "5kg", "5kilos" -> "5 kilograms"
    measurement_patterns = {
        r'(\d+\.?\d*)\s*(kg|kgs|kilos?)\b': r'\1 kilograms',
        r'(\d+\.?\d*)\s*(g|grams?)\b': r'\1 grams',
        r'(\d+\.?\d*)\s*(lb|lbs|pounds?)\b': r'\1 pounds',
        r'(\d+\.?\d*)\s*(km|kms|kilometers?)\b': r'\1 kilometers',
        r'(\d+\.?\d*)\s*(m|meters?)\b': r'\1 meters',
        r'(\d+\.?\d*)\s*(cm|centimeters?)\b': r'\1 centimeters',
        r'(\d+\.?\d*)\s*(mm|millimeters?)\b': r'\1 millimeters',
        r'(\d+\.?\d*)\s*(mi|miles?)\b': r'\1 miles',
        r'(\d+\.?\d*)\s*(ft|feet|foot)\b': r'\1 feet',
        r'(\d+\.?\d*)\s*(in|inch|inches)\b': r'\1 inches'
    }
    
    for pattern, replacement in measurement_patterns.items():
        text = re.sub(pattern, replacement, text)
    
    return text


def compute_text_quality_score(text, lang='en'):
    """
    Compute a quality score for text based on various heuristics
    
    Args:
        text (str): The text to evaluate
        lang (str): Expected language code
        
    Returns:
        float: Quality score between 0 (poor) and 1 (excellent)
    """
    if not text or pd.isna(text):
        return 0.0
    
    text = str(text)
    score = 1.0
    
    # Check text length
    if len(text) < 5:
        score *= 0.5
    elif len(text) < 20:
        score *= 0.8
    
    # Check word count
    words = text.split()
    if len(words) < 3:
        score *= 0.6
    
    # Check for repetitive characters (e.g., "aaaaa")
    for char in set(text):
        if char * 4 in text:
            score *= 0.7
            break
    
    # Check capital letter ratio
    if len(text) > 0:
        capital_ratio = sum(1 for c in text if c.isupper()) / len(text)
        if capital_ratio > 0.5:
            score *= 0.8  # Too many capital letters is usually poor quality
    
    # Check for unusual punctuation patterns
    punct_count = sum(1 for c in text if c in string.punctuation)
    if len(text) > 0 and punct_count / len(text) > 0.3:
        score *= 0.7
    
    # Check language if text is long enough
    if len(text) > 20:
        try:
            detected_lang = detect(text)
            if detected_lang != lang:
                score *= 0.6
        except LangDetectException:
            score *= 0.8
    
    # Penalize very short sentences
    sentences = sent_tokenize(text)
    avg_sent_len = np.mean([len(s) for s in sentences]) if sentences else 0
    if avg_sent_len < 15:
        score *= 0.9
    
    # Ensure score is between 0 and 1
    return max(0.0, min(1.0, score))


def compute_text_hash(text):
    """
    Compute a hash for text to identify duplicates
    
    Args:
        text (str): Text to hash
        
    Returns:
        str: Hash of the text
    """
    if not text or pd.isna(text):
        return ""
    
    # Normalize text before hashing to find near duplicates
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    
    # Create hash
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def is_near_duplicate(text1, text2, threshold=0.9):
    """
    Check if two texts are near duplicates using cosine similarity
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        threshold (float): Similarity threshold (0-1)
        
    Returns:
        bool: True if texts are near duplicates
    """
    if not text1 or not text2 or pd.isna(text1) or pd.isna(text2):
        return False
    
    # For very short texts, use direct comparison
    if len(text1) < 20 or len(text2) < 20:
        return text1.strip() == text2.strip()
    
    # For longer texts, use TF-IDF and cosine similarity
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity >= threshold
    except:
        # Fallback if vectorization fails
        return False


class PreprocessingModel(NlpModel):
    __slots__ = [
        "SAVE_DATA_DIR", "STATS_DIR", "df", "patterns", 
        "word_replacements", "lemmatizer", "stemmer", "tweet_tokenizer",
        "custom_stop_words", "use_lemmatization", "use_stemming", 
        "preserve_negation", "preserve_named_entities", "preserve_numbers",
        "correct_spelling", "advanced_tokenization", "use_spacy",
        "remove_duplicates", "min_quality_score", "technical_terms"
    ]

    def __init__(self, 
                 use_lemmatization=True, 
                 use_stemming=False,
                 preserve_negation=True,
                 preserve_named_entities=True,
                 preserve_numbers=True,
                 correct_spelling=True,
                 advanced_tokenization=True,
                 use_spacy=True,
                 remove_duplicates=True,
                 min_quality_score=0.6,
                 domain_specific_terms=None):
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
            remove_duplicates (bool): Whether to identify and remove near-duplicate texts
            min_quality_score (float): Minimum quality score for texts (0-1)
            domain_specific_terms (list): List of domain-specific terms to preserve
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
        self.remove_duplicates = remove_duplicates
        self.min_quality_score = min_quality_score
        
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
            'hashtags': re.compile(r'#\w+'),
            'repeated_chars': re.compile(r'(.)\1{3,}'),  # Repeated characters like "aaaaa"
            'emoji_pattern': re.compile(r':[a-z_]+:')
        }
        
        # Enhance word replacements dictionary
        self.word_replacements = {**SLANG_DICT, **CONTRACTIONS_DICT}
        
        # Add common technical abbreviations
        technical_abbr = {
            'api': 'application programming interface',
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'nlp': 'natural language processing',
            'db': 'database',
            'sql': 'structured query language',
            'os': 'operating system',
            'ui': 'user interface',
            'ux': 'user experience',
            'css': 'cascading style sheets',
            'html': 'hypertext markup language',
            'js': 'javascript',
            'py': 'python'
        }
        self.word_replacements.update(technical_abbr)
        
        # Add domain-specific terms to preserve
        self.technical_terms = set([
            'api', 'javascript', 'python', 'database', 'server', 'client',
            'interface', 'function', 'algorithm', 'neural', 'network',
            'data', 'machine', 'learning', 'artificial', 'intelligence',
            'framework', 'library', 'system', 'cloud', 'computing', 'model'
        ])
        
        # Add custom domain-specific terms if provided
        if domain_specific_terms:
            self.technical_terms.update([term.lower() for term in domain_specific_terms])
        
        # Define custom stop words
        self.custom_stop_words = set(stopwords.words('english'))
        
        # Remove negation words from stopwords if needed
        if self.preserve_negation:
            negation_words = {'no', 'not', 'nor', 'neither', 'never', 'none', 'wasn\'t', 'hasn\'t', 'hadn\'t', 'doesn\'t', 'don\'t', 'didn\'t', 'isn\'t', 'aren\'t', 'won\'t', 'wouldn\'t', 'couldn\'t', 'shouldn\'t'}
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
                
            # Keep technical terms intact
            if token.text.lower() in self.technical_terms:
                tokens.append(token.text.lower())
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
        
        # Apply text normalization for Unicode and special characters
        text = normalize_text(text)
        
        # Convert emojis to text
        text = emoji.demojize(text)
        
        # Clean HTML content more effectively
        text = BeautifulSoup(text, "html.parser").get_text()
        
        # Normalize numbers, dates, and measurements
        if self.preserve_numbers:
            text = normalize_numbers_and_dates(text)
        
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
        
        # Fix repeated characters (e.g., "aaaaa" -> "aa")
        text = self.patterns['repeated_chars'].sub(r'\1\1', text)
        
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
        
        # Preserve technical terms
        final_tokens = []
        for token in tokens:
            if token.lower() in self.technical_terms:
                final_tokens.append(token.lower())
                continue
                
            # Apply POS tagging for better lemmatization if needed
            if self.use_lemmatization:
                pos_tags = nltk.pos_tag([token])
                lemmatized = self.lemmatizer.lemmatize(token, get_wordnet_pos(token, pos_tags[0][1]))
                final_tokens.append(lemmatized)
            # Apply stemming if needed (not recommended with lemmatization)
            elif self.use_stemming:
                final_tokens.append(self.stemmer.stem(token))
            else:
                final_tokens.append(token)
        
        # Remove stopwords while preserving special terms
        tokens = [token for token in final_tokens 
                if token not in self.custom_stop_words]
        
        # Get sentences
        sentences = sent_tokenize(' '.join(words))
        
        # Reconstruct processed text
        processed_text = ' '.join(tokens)
        
        # Correct spelling if enabled
        if self.correct_spelling:
            try:
                processed_text = str(TextBlob(processed_text).correct())
                # Re-tokenize after spelling correction
                if self.advanced_tokenization:
                    tokens = self.tweet_tokenizer.tokenize(processed_text)
                else:
                    tokens = word_tokenize(processed_text)
            except:
                self.logger.warning("Spelling correction failed, using original tokens")
        
        # No entities in non-spaCy mode
        entities = {}
        
        return processed_text, tokens, sentences, entities

    @staticmethod
    def _process_chunk(args):
        """Process a chunk of texts in parallel"""
        
        texts, preprocessor = args
        if len(texts) == 0:
            return [], [], [], [], []
        
        if isinstance(texts, list):
            texts = pd.Series(texts)
        
        processed = np.empty(len(texts), dtype=object)
        tokenized = np.empty(len(texts), dtype=object)
        sentence_tokenized = np.empty(len(texts), dtype=object)
        entities = np.empty(len(texts), dtype=object)
        quality_scores = np.empty(len(texts), dtype=float)
        
        for i in range(len(texts)):
            text = str(texts.iloc[i])
            
            # Process text using the preprocessor's method
            processed_text, tokens, sentences, ents = preprocessor._process_text(text)
            
            # Compute quality score
            quality_score = compute_text_quality_score(processed_text)
            
            processed[i] = processed_text
            tokenized[i] = tokens
            sentence_tokenized[i] = sentences
            entities[i] = ents
            quality_scores[i] = quality_score
            
        return processed.tolist(), tokenized.tolist(), sentence_tokenized.tolist(), entities.tolist(), quality_scores.tolist()
   
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
            return [], [], [], [], []
        
    def preprocess_dataframe(self) -> None:
        """Main preprocessing pipeline with improved parallelization and quality filtering"""
        self.logger.info("PREPROCESSING STARTED..")
        self.print_section("PREPROCESSING STARTED..")
        
        if self.df is None or self.df.empty or 'Text' not in self.df.columns:
            self.logger.error("DataFrame or 'Text' column is empty. Aborting preprocessing.")
            return
            
        # Store original length for alignment
        original_length = len(self.df)
        console.print(f"[bold cyan]Processing {original_length} texts...[/bold cyan]")
        
        # Initialize empty lists for results
        processed_texts = []
        tokenized_texts = []
        sentence_tokenized_texts = []
        entities_list = []
        quality_scores = []
        
        # Determine processing approach based on data size
        if len(self.df) > 10000:
            # Adjust chunk size based on total size
            total_rows = len(self.df)
            rows_per_chunk = min(5000, max(1000, total_rows // 20))  # Dynamic chunk size
            n_chunks = max(1, int(total_rows / rows_per_chunk))
            
            n_cores = os.cpu_count() or 4
            n_cores = min(n_cores, 8)
            
            console.print(f"[bold cyan]Using {n_cores} cores to process {n_chunks} chunks...[/bold cyan]")
            
            chunks = np.array_split(self.df['Text'], n_chunks)
            process_args = [(chunk, self) for chunk in chunks]
            
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                futures = []
                for args in process_args:
                    futures.append(executor.submit(PreprocessingModel.process_with_timeout, args))
                
                # Process results with progress tracking
                with console.status("[bold green]Processing chunks...") as status:
                    for i, future in enumerate(futures, 1):
                        try:
                            chunk_results = future.result(timeout=300)
                            if len(chunk_results) == 5:
                                chunk_processed, chunk_tokens, chunk_sentences, chunk_entities, chunk_quality = chunk_results
                                processed_texts.extend(chunk_processed)
                                tokenized_texts.extend(chunk_tokens)
                                sentence_tokenized_texts.extend(chunk_sentences)
                                entities_list.extend(chunk_entities)
                                quality_scores.extend(chunk_quality)
                                console.print(f"[green]Completed chunk {i}/{n_chunks} ({(i/n_chunks)*100:.1f}%)[/green]")
                            else:
                                raise ValueError(f"Expected 5 values, got {len(chunk_results)}")
                        except Exception as e:
                            self.logger.error(f"Error processing chunk {i}/{n_chunks}: {e}")
                            console.print(f"[red]Error in chunk {i}/{n_chunks}: {e}[/red]")
                            # Add empty results for failed chunk
                            chunk_size = len(chunks[i-1])
                            processed_texts.extend([''] * chunk_size)
                            tokenized_texts.extend([[] for _ in range(chunk_size)])
                            sentence_tokenized_texts.extend([[] for _ in range(chunk_size)])
                            entities_list.extend([{} for _ in range(chunk_size)])
                            quality_scores.extend([0.0 for _ in range(chunk_size)])
        else:
            # For smaller datasets, process directly with progress tracking
            with console.status("[bold green]Processing texts...") as status:
                for i, text in enumerate(self.df['Text'], 1):
                    try:
                        processed, tokens, sentences, entities = self._process_text(text)
                        quality = compute_text_quality_score(processed)
                        processed_texts.append(processed)
                        tokenized_texts.append(tokens)
                        sentence_tokenized_texts.append(sentences)
                        entities_list.append(entities)
                        quality_scores.append(quality)
                        
                        if i % 100 == 0:
                            console.print(f"[green]Processed {i}/{len(self.df)} texts ({(i/len(self.df))*100:.1f}%)[/green]")
                    except Exception as e:
                        self.logger.error(f"Error processing text {i}/{len(self.df)}: {e}")
                        processed_texts.append('')
                        tokenized_texts.append([])
                        sentence_tokenized_texts.append([])
                        entities_list.append({})
                        quality_scores.append(0.0)
        
        console.print("[bold cyan]Finalizing results...[/bold cyan]")
        
        # Ensure all lists have the same length as the original DataFrame
        def pad_list(lst, target_length, default_value):
            if len(lst) < target_length:
                lst.extend([default_value for _ in range(target_length - len(lst))])
            return lst[:target_length]
        
        processed_texts = pad_list(processed_texts, original_length, '')
        tokenized_texts = pad_list(tokenized_texts, original_length, [])
        sentence_tokenized_texts = pad_list(sentence_tokenized_texts, original_length, [])
        entities_list = pad_list(entities_list, original_length, {})
        quality_scores = pad_list(quality_scores, original_length, 0.0)
        
        # Create new DataFrame columns
        new_data = {
            'Text': processed_texts,
            'Tokens': tokenized_texts,
            'Sentences': sentence_tokenized_texts,
            'Entities': entities_list,
            'Token_count': [len(tokens) for tokens in tokenized_texts],
            'Sentence_count': [len(sentences) for sentences in sentence_tokenized_texts],
            'Text_length': [len(text) for text in processed_texts],
            'Quality_score': quality_scores
        }
        
        # Add sentiment analysis if TextBlob is available
        try:
            console.print("[bold cyan]Adding sentiment analysis...[/bold cyan]")
            sentiment_scores = [TextBlob(text).sentiment.polarity for text in processed_texts]
            subjectivity_scores = [TextBlob(text).sentiment.subjectivity for text in processed_texts]
            new_data['Sentiment'] = sentiment_scores
            new_data['Subjectivity'] = subjectivity_scores
        except Exception as e:
            self.logger.warning(f"TextBlob sentiment analysis failed: {e}")
        
        # Create new DataFrame with proper length alignment
        console.print("[bold cyan]Creating final DataFrame...[/bold cyan]")
        result_df = pd.DataFrame(new_data, index=range(original_length))
        
        # Preserve original columns if they exist
        preserve_columns = ['Id', 'Score', 'Summary'] if all(col in self.df.columns for col in ['Id', 'Score', 'Summary']) else []
        if preserve_columns:
            for col in preserve_columns:
                result_df[col] = self.df[col].values
        
        self.df = result_df
        
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
            except LangDetectException:
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
        """Perform advanced feature engineering on preprocessed data"""
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
        
        # Text complexity features
        try:
            # Lexical diversity (unique words / total words)
            self.df['Lexical_diversity'] = self.df['Tokens'].apply(
                lambda x: len(set(x)) / len(x) if x and len(x) > 0 else 0
            )
            
            # Named entity count
            if 'Entities' in self.df.columns:
                self.df['Entity_count'] = self.df['Entities'].apply(len)
                
            # Create POS tag distributions
            if self.use_spacy:
                self.logger.info("Generating POS tag distributions")
                
                def get_pos_distribution(text):
                    if not text or pd.isna(text):
                        return {}
                    
                    doc = nlp(str(text))
                    pos_counts = Counter([token.pos_ for token in doc])
                    total = sum(pos_counts.values())
                    
                    if total == 0:
                        return {}
                    
                    return {pos: count/total for pos, count in pos_counts.items()}
                
                # Apply to a sample of texts to avoid memory issues
                sample_size = min(1000, len(self.df))
                sample_idx = np.random.choice(len(self.df), sample_size, replace=False)
                
                for i in sample_idx:
                    pos_dist = get_pos_distribution(self.df.iloc[i]['Text'])
                    
                    # Add POS distribution features
                    for pos, value in pos_dist.items():
                        col_name = f'POS_{pos}'
                        if col_name not in self.df.columns:
                            self.df[col_name] = 0.0
                        self.df.at[i, col_name] = value
            
            # Extract n-grams
            self.logger.info("Extracting n-grams")
            
            def extract_ngrams(tokens, n=2):
                if not tokens or len(tokens) < n:
                    return []
                return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            
            # Extract bigrams and trigrams
            self.df['Bigrams'] = self.df['Tokens'].apply(lambda x: extract_ngrams(x, 2))
            self.df['Trigrams'] = self.df['Tokens'].apply(lambda x: extract_ngrams(x, 3))
            
        except Exception as e:
            self.logger.warning(f"Some feature engineering steps failed: {e}")
        
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
        summary.add_row("Remove Duplicates", str(self.remove_duplicates))
        summary.add_row("Min Quality Score", str(self.min_quality_score))
        
        # Add processing results if available
        if self.df is not None and not self.df.empty:
            summary.add_row("Processed Rows", str(len(self.df)))
            if 'Token_count' in self.df.columns:
                summary.add_row("Avg Token Count", f"{self.df['Token_count'].mean():.2f}")
            if 'Sentence_count' in self.df.columns:
                summary.add_row("Avg Sentence Count", f"{self.df['Sentence_count'].mean():.2f}")
            if 'Quality_score' in self.df.columns:
                summary.add_row("Avg Quality Score", f"{self.df['Quality_score'].mean():.3f}")
            if 'Sentiment' in self.df.columns:
                summary.add_row("Avg Sentiment", f"{self.df['Sentiment'].mean():.3f}")
            if 'Lexical_diversity' in self.df.columns:
                summary.add_row("Avg Lexical Diversity", f"{self.df['Lexical_diversity'].mean():.3f}")
        
        console.print(summary)
        
    def find_similar_texts(self, threshold=0.8, sample_size=1000):
        """
        Find similar texts in the dataset using TF-IDF and cosine similarity
        
        Args:
            threshold (float): Similarity threshold (0-1)
            sample_size (int): Number of texts to sample for comparison
            
        Returns:
            pd.DataFrame: DataFrame with similar text pairs
        """
        if self.df is None or self.df.empty:
            self.logger.error("DataFrame is empty. Cannot find similar texts.")
            return pd.DataFrame()
            
        self.logger.info(f"Finding similar texts with threshold {threshold}")
        console.print(f"[bold cyan]Finding similar texts with threshold {threshold}...[/bold cyan]")
        
        # Sample texts to keep computation manageable
        sample_size = min(sample_size, len(self.df))
        sample_indices = np.random.choice(len(self.df), sample_size, replace=False)
        sample_texts = self.df.iloc[sample_indices]['Text'].tolist()
        
        # Vectorize texts
        vectorizer = TfidfVectorizer(min_df=2, max_df=0.95)
        try:
            tfidf_matrix = vectorizer.fit_transform(sample_texts)
        except:
            self.logger.error("Vectorization failed. Check text content.")
            return pd.DataFrame()
        
        # Compute pairwise cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find similar pairs
        similar_pairs = []
        for i in range(len(sample_texts)):
            for j in range(i+1, len(sample_texts)):
                if similarity_matrix[i, j] >= threshold:
                    similar_pairs.append({
                        'Index1': sample_indices[i],
                        'Index2': sample_indices[j],
                        'Text1': sample_texts[i],
                        'Text2': sample_texts[j],
                        'Similarity': similarity_matrix[i, j]
                    })
        
        if not similar_pairs:
            self.logger.info("No similar text pairs found.")
            console.print("[bold yellow]No similar text pairs found.[/bold yellow]")
            return pd.DataFrame()
        
        similar_df = pd.DataFrame(similar_pairs)
        self.logger.info(f"Found {len(similar_df)} similar text pairs.")
        console.print(f"[bold green]Found {len(similar_df)} similar text pairs.[/bold green]")
        
        return similar_df

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
            use_spacy=True,  # Set to False for faster but less accurate processing
            remove_duplicates=True,
            min_quality_score=0.6,
            domain_specific_terms=['nlp', 'dataset', 'preprocessing', 'tokenization', 'sentiment']
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

        # Remove non-English text
        model.remove_foreign_words()
        
        # Perform feature engineering
        model.perform_feature_engineering()
        
        # Find similar texts
        similar_texts = model.find_similar_texts(threshold=0.85, sample_size=500)
        if not similar_texts.empty:
            console.print("[bold cyan]Sample of similar text pairs:[/bold cyan]")
            console.print(similar_texts.head(3))
        
        # Save to multiple formats
        model.save_to_csv(output_path=OUTPUT_CSV)
        model.save_to_parquet(output_path=OUTPUT_PARQUET)
        
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