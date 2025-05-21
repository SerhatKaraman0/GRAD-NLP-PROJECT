"""
GPU-optimized NLP preprocessing model with enhanced CUDA operations.
Implements efficient text processing using GPU acceleration.
"""
import os
import sys
import re
import gc
import string
import hashlib
import unicodedata
import itertools
import timeit
import cProfile
import pstats
from collections import Counter
from functools import partial

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords, wordnet

from textblob import TextBlob
from bs4 import BeautifulSoup
import spacy
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import emoji

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dateutil import parser as date_parser

from rich.console import Console
from rich.table import Table
from rich.progress import track

import swifter

from src.core.nlpmodel import NlpModel
from src.core.logging_config import *  # noqa: F403, F405
from utils.helper import CONTRACTIONS_DICT, SLANG_DICT
from src.data.cuda_text_kernels import CudaTextOperations, batch_process_with_cuda, CudaTextEmbedding
from src.data.gpu_memory_manager import GpuMemoryTracker, clear_gpu_memory, get_gpu_memory_status
from src.data.gpu_parallel_processor import GpuParallelProcessor, ParallelTextDataset
from src.data.fix_length_mismatch import fix_dataframe_length_mismatch

# GPU-specific imports
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import torch.nn as nn

# Import enhanced GPU modules
from src.data.cuda_text_kernels import CudaTextOperations, batch_process_with_cuda, CudaTextEmbedding
from src.data.gpu_memory_manager import GpuMemoryTracker, clear_gpu_memory, get_gpu_memory_status
from src.data.gpu_parallel_processor import GpuParallelProcessor, ParallelTextDataset

# Set seed for language detection to ensure consistent results
DetectorFactory.seed = 42

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


# GPU-specific helper functions
def get_device():
    """Get the best available device (GPU if available, otherwise CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        console.print(f"[bold green]Using GPU: {torch.cuda.get_device_name(0)}[/bold green]")
    else:
        device = torch.device("cpu")
        console.print("[bold yellow]GPU not available, using CPU[/bold yellow]")
    return device


def pad_sequences(sequences, max_len=None, padding_value=0):
    """
    Pad sequences to the same length for batch processing on GPU
    
    Args:
        sequences (list): List of sequences (lists or arrays)
        max_len (int): Maximum length to pad to (None for auto)
        padding_value (int): Value to pad with
        
    Returns:
        torch.Tensor: Padded tensor
    """
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    padded_seqs = []
    for seq in sequences:
        seq_len = len(seq)
        if seq_len < max_len:
            # Pad the sequence
            padded = list(seq) + [padding_value] * (max_len - seq_len)
            padded_seqs.append(padded)
        else:
            # Truncate if longer than max_len
            padded_seqs.append(seq[:max_len])
    
    return torch.tensor(padded_seqs)


def create_data_loader(texts, batch_size=32):
    """Create a PyTorch DataLoader for text processing"""
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    dataset = TextDataset(texts)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)


class TextDataset(Dataset):
    """Simple PyTorch Dataset for text data"""
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]


class GpuTextProcessor(nn.Module):
    """
    PyTorch module for text processing operations
    This enables parallel processing on GPU when possible
    """
    def __init__(self):
        super(GpuTextProcessor, self).__init__()
        # No learnable parameters, just a container for operations
    
    def forward(self, texts):
        # This is just a placeholder
        # Actual processing is done in specific methods
        return texts
    
    def normalize_texts(self, texts):
        """Normalize a batch of texts"""
        if isinstance(texts, torch.Tensor):
            texts = texts.tolist()
        
        results = []
        for text in texts:
            results.append(normalize_text(text))
        
        return results
    
    def process_batch(self, texts, preserve_numbers=True, correct_spelling=True):
        """Process a batch of texts in parallel"""
        normalized = self.normalize_texts(texts)
        
        processed_texts = []
        tokens_list = []
        sentences_list = []
        entities_list = []
        
        for text in normalized:
            # Process each normalized text using CPU functions (for now)
            # Future: Implement more GPU operations using CUDA kernels
            processed_text, tokens, sentences, entities = self._process_single_text(
                text,
                preserve_numbers=preserve_numbers,
                correct_spelling=correct_spelling
            )
            
            processed_texts.append(processed_text)
            tokens_list.append(tokens)
            sentences_list.append(sentences)
            entities_list.append(entities)
        
        return processed_texts, tokens_list, sentences_list, entities_list
    
    def _process_single_text(self, text, preserve_numbers=True, correct_spelling=True):
        """Process a single text (will be used until full GPU implementation)"""
        if not text or pd.isna(text):
            return "", [], [], {}
        
        # Convert emojis to text
        text = emoji.demojize(text)
        
        # Clean HTML content
        text = BeautifulSoup(text, "html.parser").get_text()
        
        # Normalize numbers, dates, and measurements
        if preserve_numbers:
            text = normalize_numbers_and_dates(text)
        
        # Apply regex patterns (still CPU-based)
        # Remove URLs, emails, user mentions
        text = re.sub(r'(https?|ftp):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])', ' ', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)
        text = re.sub(r'@\w+', ' ', text)
        
        # Handle numbers based on configuration
        if not preserve_numbers:
            text = re.sub(r'\b\d+\b', ' ', text)
        
        # Remove special characters but handle hashtags differently
        hashtags = re.findall(r'#\w+', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Add back hashtags without the # symbol if they represent meaningful content
        for tag in hashtags:
            text += f" {tag[1:]} "
        
        # Fix repeated characters (e.g., "aaaaa" -> "aa")
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)
        
        # Normalize whitespace
        text = " ".join(text.split())
        
        # Use spaCy for NLP processing
        doc = nlp(text)
        
        # Extract named entities
        entities = {ent.text: ent.label_ for ent in doc.ents}
        
        # Get tokens
        tokens = []
        for token in doc:
            # Skip stop words
            if token.is_stop:
                continue
            
            # Include lemma
            tokens.append(token.lemma_)
        
        # Get sentences
        sentences = [sent.text for sent in doc.sents]
        
        # Reconstruct processed text
        processed_text = " ".join(tokens)
        
        # Correct spelling if enabled
        if correct_spelling:
            try:
                processed_text = str(TextBlob(processed_text).correct())
                # Re-tokenize after spelling correction
                tokens = word_tokenize(processed_text)
            except:
                pass
        
        return processed_text, tokens, sentences, entities


class PreprocessingModelGPU(NlpModel):
    __slots__ = [
        "SAVE_DATA_DIR", "STATS_DIR", "df", "patterns", 
        "word_replacements", "lemmatizer", "stemmer", "tweet_tokenizer",
        "custom_stop_words", "use_lemmatization", "use_stemming", 
        "preserve_negation", "preserve_named_entities", "preserve_numbers",
        "correct_spelling", "advanced_tokenization", "use_spacy",
        "remove_duplicates", "min_quality_score", "technical_terms",
        "use_gpu", "device", "batch_size", "gpu_processor", "max_seq_len"
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
                 domain_specific_terms=None,
                 use_gpu=True,
                 batch_size=32,
                 max_seq_len=None,
                 memory_threshold=80.0,
                 num_workers=4,
                 advanced_cuda=True):
        """
        Initialize the GPU-optimized preprocessing model with configurable options
        
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
            use_gpu (bool): Whether to use GPU for processing (if available)
            batch_size (int): Batch size for GPU processing
            max_seq_len (int): Maximum sequence length for padding (None for auto)
            memory_threshold (float): GPU memory threshold percentage
            num_workers (int): Number of workers for parallel processing
            advanced_cuda (bool): Whether to use advanced CUDA operations
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
        
        # GPU-specific settings
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.memory_threshold = memory_threshold
        self.num_workers = num_workers
        self.advanced_cuda = advanced_cuda
        
        # Initialize memory tracker
        self.memory_tracker = GpuMemoryTracker(threshold_percent=memory_threshold)
        
        # Initialize GPU processors
        if self.use_gpu and torch.cuda.is_available():
            # Use legacy processor for backward compatibility
            self.gpu_processor = GpuTextProcessor().to(self.device)
            
            # Use advanced processors if enabled
            if self.advanced_cuda:
                self.cuda_ops = CudaTextOperations(device=self.device)
                self.parallel_processor = GpuParallelProcessor(
                    batch_size=batch_size,
                    num_workers=num_workers,
                    device=self.device,
                    memory_threshold=memory_threshold
                )
                console.print("[bold green]Initialized advanced CUDA operations[/bold green]")
        else:
            self.gpu_processor = None
            self.cuda_ops = None
            self.parallel_processor = None
        
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
    
    def preprocess_dataframe(self) -> None:
        """
        Main preprocessing pipeline optimized for GPU execution
        Handles the length mismatch error by ensuring all result lists have the same length
        """
        self.logger.info("GPU-OPTIMIZED PREPROCESSING STARTED..")
        self.print_section("GPU-OPTIMIZED PREPROCESSING STARTED..")
        
        if self.df is None or self.df.empty or 'Text' not in self.df.columns:
            self.logger.error("DataFrame or 'Text' column is empty. Aborting preprocessing.")
            return
        
        # Calculate original dataframe length for proper alignment later
        original_length = len(self.df)
        
        # Step 1: Calculate text hashes for duplicate detection if enabled
        if self.remove_duplicates:
            self.logger.info("Calculating text hashes for duplicate detection")
            console.print("[bold cyan]Calculating text hashes for duplicate detection...[/bold cyan]")
            
            self.df['Text_hash'] = self.df['Text'].apply(compute_text_hash)
            
            # Remove exact duplicates
            hash_counts = self.df['Text_hash'].value_counts()
            duplicated_hashes = hash_counts[hash_counts > 1].index.tolist()
            
            if duplicated_hashes:
                dup_mask = self.df['Text_hash'].isin(duplicated_hashes)
                dup_count = dup_mask.sum()
                self.logger.info(f"Found {dup_count} exact duplicates")
                console.print(f"[bold yellow]Found {dup_count} exact duplicates[/bold yellow]")
                
                # Keep only the first occurrence of each duplicate
                self.df = self.df.drop_duplicates(subset='Text_hash')
        
        # Step 2: GPU-based text processing in batches
        if self.use_gpu and torch.cuda.is_available():
            self.logger.info("Using GPU for preprocessing")
            console.print("[bold green]Using GPU for preprocessing...[/bold green]")
            
            data_loader = create_data_loader(self.df['Text'], batch_size=self.batch_size)
            
            # Initialize result containers
            all_processed_texts = []
            all_tokenized_texts = []
            all_sentence_tokenized_texts = []
            all_entities_list = []
            
            # Process in batches
            for batch in track(data_loader, description="Processing batches..."):
                processed_texts, tokenized_texts, sentence_tokenized_texts, entities_list = self.gpu_processor.process_batch(
                    batch,
                    preserve_numbers=self.preserve_numbers,
                    correct_spelling=self.correct_spelling
                )
                
                all_processed_texts.extend(processed_texts)
                all_tokenized_texts.extend(tokenized_texts)
                all_sentence_tokenized_texts.extend(sentence_tokenized_texts)
                all_entities_list.extend(entities_list)
                
                # Force garbage collection to free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # Calculate quality scores
            quality_scores = [compute_text_quality_score(text) for text in all_processed_texts]
            
            # Create new dataframe for results
            new_data = {
                'Text': all_processed_texts,
                'Tokens': all_tokenized_texts,
                'Sentences': all_sentence_tokenized_texts,
                'Entities': all_entities_list,
                'Token_count': [len(tokens) for tokens in all_tokenized_texts],
                'Sentence_count': [len(sentences) for sentences in all_sentence_tokenized_texts],
                'Text_length': [len(text) for text in all_processed_texts],
                'Quality_score': quality_scores
            }
            
            # Add sentiment analysis if TextBlob is available
            try:
                sentiment_scores = [TextBlob(text).sentiment.polarity for text in all_processed_texts]
                subjectivity_scores = [TextBlob(text).sentiment.subjectivity for text in all_processed_texts]
                new_data['Sentiment'] = sentiment_scores
                new_data['Subjectivity'] = subjectivity_scores
            except:
                self.logger.warning("TextBlob sentiment analysis failed, skipping sentiment features")
            
            # Ensure all result arrays have the same length
            for key, value in new_data.items():
                if len(value) != len(all_processed_texts):
                    self.logger.warning(f"Length mismatch in {key}: {len(value)} vs {len(all_processed_texts)}")
                    # Pad or truncate to ensure length match
                    if len(value) < len(all_processed_texts):
                        if isinstance(value[0], (list, tuple, dict)):
                            value.extend([[] if isinstance(value[0], (list, tuple)) else {} for _ in range(len(all_processed_texts) - len(value))])
                        else:
                            value.extend([0 for _ in range(len(all_processed_texts) - len(value))])
                    else:
                        value = value[:len(all_processed_texts)]
                    new_data[key] = value
            
            # Create result dataframe with proper index alignment
            result_df = pd.DataFrame(new_data)
            
            # Ensure result_df has the same length as the original dataframe
            # This is critical to avoid the 'Length of values does not match length of index' error
            if len(result_df) != original_length:
                self.logger.warning(f"Length mismatch between result ({len(result_df)}) and original ({original_length})")
                
                # Create a DataFrame with the same index as the original
                aligned_df = pd.DataFrame(index=range(original_length))
                
                # Add preserve columns first if they exist
                preserve_columns = ['Id', 'Score', 'Summary'] if all(col in self.df.columns for col in ['Id', 'Score', 'Summary']) else []
                if preserve_columns:
                    for col in preserve_columns:
                        aligned_df[col] = self.df[col].values
                
                # Add processed data columns with proper padding/truncation
                for col in result_df.columns:
                    if col in preserve_columns:
                        continue  # Already added
                        
                    # Get the data to assign
                    data = result_df[col].values
                    
                    # Handle length mismatch
                    if len(data) < original_length:
                        # Need to pad with default values
                        if col in ['Tokens', 'Sentences']:
                            padding = [[] for _ in range(original_length - len(data))]
                        elif col == 'Entities':
                            padding = [{} for _ in range(original_length - len(data))]
                        elif col in ['Quality_score', 'Sentiment', 'Subjectivity']:
                            padding = [0.0 for _ in range(original_length - len(data))]
                        elif col in ['Token_count', 'Sentence_count', 'Text_length']:
                            padding = [0 for _ in range(original_length - len(data))]
                        else:
                            padding = [''] * (original_length - len(data))
                            
                        # Extend the data with padding
                        data = list(data) + padding
                    elif len(data) > original_length:
                        # Need to truncate
                        data = data[:original_length]
                    
                    # Assign to the aligned dataframe
                    aligned_df[col] = data
                    
                # Use the aligned dataframe as our result
                result_df = aligned_df
            
            # Preserve original columns if needed
            preserve_columns = ['Id', 'Score', 'Summary'] if all(col in self.df.columns for col in ['Id', 'Score', 'Summary']) else []
            
            if preserve_columns:
                # Create a new dataframe with original index and columns
                new_df = self.df[preserve_columns].copy()
                
                # Add the processed data columns
                for col in result_df.columns:
                    new_df[col] = result_df[col].values
                
                self.df = new_df
            else:
                # Just use the result dataframe
                self.df = result_df
        else:
            # Fallback to CPU processing if GPU is not available
            self.logger.info("GPU not available, using CPU for preprocessing")
            console.print("[bold yellow]GPU not available, using CPU for preprocessing...[/bold yellow]")
            
            # Process texts using swifter for parallel CPU processing
            results = self.df['Text'].swifter.progress_bar(True).apply(self._process_text)
            
            # Unpack results safely
            processed_texts, tokenized_texts, sentence_tokenized_texts, entities_list = [], [], [], []
            
            for result in results:
                if len(result) == 4:  # Ensure each result has all components
                    pt, tt, st, el = result
                    processed_texts.append(pt)
                    tokenized_texts.append(tt)
                    sentence_tokenized_texts.append(st)
                    entities_list.append(el)
                else:
                    # Add empty values for missing results
                    processed_texts.append("")
                    tokenized_texts.append([])
                    sentence_tokenized_texts.append([])
                    entities_list.append({})
            
            # Calculate quality scores
            quality_scores = [compute_text_quality_score(text) for text in processed_texts]
            
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
                sentiment_scores = [TextBlob(text).sentiment.polarity for text in processed_texts]
                subjectivity_scores = [TextBlob(text).sentiment.subjectivity for text in processed_texts]
                new_data['Sentiment'] = sentiment_scores
                new_data['Subjectivity'] = subjectivity_scores
            except:
                self.logger.warning("TextBlob sentiment analysis failed, skipping sentiment features")
            
            # Preserve original columns and add new ones
            preserve_columns = ['Id', 'Score', 'Summary'] if all(col in self.df.columns for col in ['Id', 'Score', 'Summary']) else []
            self.df = self.df[preserve_columns].assign(**new_data) if preserve_columns else pd.DataFrame(new_data)
        
        # Filter out low-quality texts
        if self.min_quality_score > 0:
            low_quality_count = (self.df['Quality_score'] < self.min_quality_score).sum()
            self.logger.info(f"Filtering out {low_quality_count} low-quality texts")
            console.print(f"[bold yellow]Filtering out {low_quality_count} low-quality texts[/bold yellow]")
            
            # Replace low-quality texts with empty string
            self.df.loc[self.df['Quality_score'] < self.min_quality_score, 'Text'] = ''
            
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("GPU-optimized preprocessing completed successfully")
        console.print("[bold green]GPU-optimized preprocessing completed successfully[/bold green]")

    def _process_text(self, text):
        """Process a single text - CPU fallback method"""
        
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
        self.df['Word_count'] = self.df['Tokens'].apply(len)
        
        if 'Sentences' in self.df.columns:
            self.df['Sentence_count'] = self.df['Sentences'].apply(len)
            self.df['Avg_sentence_length'] = self.df.apply(
                lambda x: np.mean([len(s) for s in x['Sentences']]) if x['Sentences'] else 0, 
                axis=1
            )
        
        # Compute complexity metrics
        if 'Tokens' in self.df.columns:
            # Word length
            self.df['Avg_word_length'] = self.df['Tokens'].apply(
                lambda x: np.mean([len(w) for w in x]) if x else 0
            )
            
            # Vocabulary richness (unique words ratio)
            self.df['Vocabulary_richness'] = self.df['Tokens'].apply(
                lambda x: len(set(x)) / len(x) if x else 0
            )
        
        # Create entity features if available
        if 'Entities' in self.df.columns:
            self.df['Entity_count'] = self.df['Entities'].apply(len)
            
            # Extract entity types
            entity_types = set()
            for entities in self.df['Entities']:
                entity_types.update(entities.values())
            
            # Create features for entity types
            for entity_type in entity_types:
                col_name = f'Entity_{entity_type}'
                self.df[col_name] = self.df['Entities'].apply(
                    lambda x: sum(1 for t in x.values() if t == entity_type)
                )
        
        self.logger.info("Feature engineering completed")
        console.print("[bold green]Feature engineering completed[/bold green]")

    def get_preprocessing_summary(self):
        """Generate and print a summary of preprocessing results"""
        if self.df is None or self.df.empty:
            self.logger.error("DataFrame is empty. Cannot generate summary.")
            return
            
        self.logger.info("Generating preprocessing summary")
        self.print_section("PREPROCESSING SUMMARY")
        
        summary = Table(title="Preprocessing Summary", show_header=True, header_style="bold magenta")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="green")
        
        # Get basic stats
        total_texts = len(self.df)
        empty_texts = (self.df['Text'] == '').sum()
        avg_tokens = self.df['Tokens'].apply(len).mean()
        avg_token_len = self.df['Tokens'].apply(lambda x: np.mean([len(t) for t in x]) if x else 0).mean()
        
        # Quality metrics
        if 'Quality_score' in self.df.columns:
            avg_quality = self.df['Quality_score'].mean()
            low_quality = (self.df['Quality_score'] < self.min_quality_score).sum();
            
            summary.add_row("Average Quality Score", f"{avg_quality:.4f}")
            summary.add_row("Low Quality Texts", f"{low_quality} ({low_quality/total_texts*100:.2f}%)")
        
        # Sentiment metrics
        if 'Sentiment' in self.df.columns:
            avg_sentiment = self.df['Sentiment'].mean()
            positive_texts = (self.df['Sentiment'] > 0.05).sum()
            negative_texts = (self.df['Sentiment'] < -0.05).sum()
            neutral_texts = total_texts - positive_texts - negative_texts;
            
            summary.add_row("Average Sentiment", f"{avg_sentiment:.4f}")
            summary.add_row("Positive Texts", f"{positive_texts} ({positive_texts/total_texts*100:.2f}%)")
            summary.add_row("Negative Texts", f"{negative_texts} ({negative_texts/total_texts*100:.2f}%)")
            summary.add_row("Neutral Texts", f"{neutral_texts} ({neutral_texts/total_texts*100:.2f}%)")
        
        # Token stats
        summary.add_row("Total Texts", str(total_texts))
        summary.add_row("Empty Texts", f"{empty_texts} ({empty_texts/total_texts*100:.2f}%)")
        summary.add_row("Average Tokens per Text", f"{avg_tokens:.2f}")
        summary.add_row("Average Token Length", f"{avg_token_len:.2f}")
        
        console.print(summary)


if __name__ == "__main__":
    with cProfile.Profile() as profile:
        # Create GPU-optimized preprocessing model with enhanced options
        model = PreprocessingModelGPU(
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
            domain_specific_terms=['nlp', 'dataset', 'preprocessing', 'tokenization', 'sentiment'],
            use_gpu=True,
            batch_size=64
        )
        
        console.print("[bold cyan]DataFrame shape:[/bold cyan]", model.df.shape)
        console.print("[bold cyan]First few rows of 'Text' column:[/bold cyan]")
        console.print(model.df['Text'].head())
        
        model.get_statistics(model.df)

        # Output paths for different formats
        OUTPUT_CSV = os.path.join(model.SAVE_DATA_DIR, "PREPROCESSED_GPU_Reviews.csv")
        OUTPUT_PARQUET = os.path.join(model.SAVE_DATA_DIR, "PREPROCESSED_GPU_Reviews.parquet")

        # Preprocess data
        start_time = timeit.default_timer()
        model.preprocess_dataframe()
        elapsed = timeit.default_timer() - start_time

        # Remove non-English text
        model.remove_foreign_words()
        
        # Perform feature engineering
        model.perform_feature_engineering()
        
        # Save to multiple formats
        model.save_to_csv(output_path=OUTPUT_CSV)
        # model.save_to_parquet(output_path=OUTPUT_PARQUET)
        
        # Load processed data and get statistics
        cleaned_df = pd.read_csv(OUTPUT_CSV)
        model.get_statistics(cleaned_df)
        
        # Print processing summary
        model.get_preprocessing_summary()

        console.print(f"[bold green]Preprocessing took {elapsed:.2f} seconds[/bold green]")
        console.print("[bold cyan]Processed DataFrame:[/bold cyan]")
        console.print(model.df.head())
        console.print("\n[bold cyan]DataFrame columns:[/bold cyan]", model.df.columns.tolist())

    stats_file_dir = os.path.join(model.STATS_DIR, "gpu_results.prof")
    
    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)

    results.dump_stats(stats_file_dir)
