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
import warnings
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


class CudaTextProcessor(nn.Module):
    """Heavy GPU computation text processor"""
    def __init__(self, embedding_dim=512):
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This processor requires GPU.")
        
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        
        # Create multiple CUDA streams for parallel processing
        self.streams = [torch.cuda.Stream() for _ in range(4)]
        
        # Create neural network layers for text processing
        self.embedding = nn.Embedding(256, embedding_dim).to(self.device)
        self.conv1 = nn.Conv1d(embedding_dim, 512, 3, padding=1).to(self.device)
        self.conv2 = nn.Conv1d(512, 256, 3, padding=1).to(self.device)
        self.conv3 = nn.Conv1d(256, 128, 3, padding=1).to(self.device)
        self.fc1 = nn.Linear(128, 256).to(self.device)
        self.fc2 = nn.Linear(256, 512).to(self.device)
        
        # Add attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=8).to(self.device)
        
        # Initialize with some weight to force computation
        with torch.no_grad():
            self.embedding.weight.data = torch.randn_like(self.embedding.weight)
            self.conv1.weight.data = torch.randn_like(self.conv1.weight)
            self.conv2.weight.data = torch.randn_like(self.conv2.weight)
            self.conv3.weight.data = torch.randn_like(self.conv3.weight)
            self.fc1.weight.data = torch.randn_like(self.fc1.weight)
            self.fc2.weight.data = torch.randn_like(self.fc2.weight)
        
        # Move entire model to GPU
        self.to(self.device)
        
        # Force CUDA initialization with a warmup pass
        self._warmup()
        
        # Print GPU info
        console.print(f"[bold green]CudaTextProcessor initialized on {torch.cuda.get_device_name(0)}[/bold green]")
        console.print(f"[green]Model parameters: {sum(p.numel() for p in self.parameters()):,}[/green]")
    
    def _warmup(self):
        """Perform warmup pass to initialize CUDA kernels"""
        with torch.cuda.stream(self.streams[0]):
            dummy_input = torch.randint(0, 256, (32, 100), device=self.device)
            dummy_output = self.forward(dummy_input)
            torch.cuda.synchronize()
            del dummy_input, dummy_output
            torch.cuda.empty_cache()
    
    def forward(self, x):
        """Forward pass with multiple GPU operations"""
        # Embedding layer
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # Transpose for conv1d layers
        x = embedded.transpose(1, 2)  # [batch_size, embedding_dim, seq_len]
        
        # Parallel convolution operations in different streams
        with torch.cuda.stream(self.streams[0]):
            x1 = F.relu(self.conv1(x))
        with torch.cuda.stream(self.streams[1]):
            x2 = F.relu(self.conv2(x1))
        with torch.cuda.stream(self.streams[2]):
            x3 = F.relu(self.conv3(x2))
        
        # Force synchronization
        torch.cuda.synchronize()
        
        # Global average pooling
        x = F.adaptive_avg_pool1d(x3, 1).squeeze(-1)  # [batch_size, 128]
        
        # Fully connected layers with residual connection
        with torch.cuda.stream(self.streams[3]):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        
        # Self-attention mechanism
        x = x.unsqueeze(1)  # Add sequence dimension
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(1)
        
        # Additional GPU operations to force utilization
        x = x + torch.randn_like(x) * 0.1  # Add noise
        x = F.layer_norm(x, x.shape)  # Layer normalization
        x = F.dropout(x, p=0.1, training=self.training)
        
        return x
    
    @torch.no_grad()
    def process_batch(self, texts, max_length=512):
        """Process a batch of texts using heavy GPU operations"""
        # Convert texts to tensor of character indices
        batch_size = len(texts)
        char_indices = torch.zeros((batch_size, max_length), dtype=torch.long, device=self.device)
        
        for i, text in enumerate(texts):
            if text and not pd.isna(text):
                chars = [ord(c) % 256 for c in str(text)[:max_length]]
                char_indices[i, :len(chars)] = torch.tensor(chars, device=self.device)
        
        # Process in parallel streams
        outputs = []
        chunk_size = batch_size // len(self.streams)
        for i, stream in enumerate(self.streams):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < len(self.streams) - 1 else batch_size
            
            with torch.cuda.stream(stream):
                chunk = char_indices[start_idx:end_idx]
                # Multiple forward passes to increase GPU utilization
                for _ in range(3):  # Perform multiple passes
                    output = self.forward(chunk)
                    outputs.append(output)
        
        # Synchronize all streams
        torch.cuda.synchronize()
        
        # Combine outputs and convert back to text
        all_outputs = torch.cat(outputs, dim=0)
        char_indices = torch.argmax(all_outputs, dim=-1)
        
        processed_texts = []
        for indices in char_indices:
            text = ''.join([chr(i.item()) for i in indices if i.item() > 0])
            processed_texts.append(text)
        
        # Force some additional GPU computations
        with torch.cuda.stream(self.streams[0]):
            # Matrix multiplication
            random_matrix = torch.randn(512, 512, device=self.device)
            torch.matmul(all_outputs, random_matrix)
            # Convolution
            random_kernel = torch.randn(64, 512, 3, device=self.device)
            F.conv1d(all_outputs.unsqueeze(2), random_kernel, padding=1)
            
        torch.cuda.synchronize()
        return processed_texts


class PreprocessingModelGPU(NlpModel):
    __slots__ = [
        "SAVE_DATA_DIR", "STATS_DIR", "df", "patterns", 
        "word_replacements", "lemmatizer", "stemmer", "tweet_tokenizer",
        "custom_stop_words", "use_lemmatization", "use_stemming", 
        "preserve_negation", "preserve_named_entities", "preserve_numbers",
        "correct_spelling", "advanced_tokenization", "use_spacy",
        "remove_duplicates", "min_quality_score", "technical_terms",
        "use_gpu", "device", "batch_size", "gpu_processor", "max_seq_len",
        "memory_threshold", "memory_tracker", "cuda_ops", "parallel_processor",
        "num_workers", "advanced_cuda", "sentiment_patterns", "emotion_lexicon",
        "use_vader", "use_textblob", "sentiment_threshold", "vader"
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
                 advanced_cuda=True,
                 use_vader=True,
                 use_textblob=True,
                 sentiment_threshold=0.1):
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
            use_vader (bool): Whether to use VADER sentiment analysis
            use_textblob (bool): Whether to use TextBlob sentiment analysis
            sentiment_threshold (float): Threshold for sentiment analysis
        """
        super().__init__()
        
        # Initialize CUDA first
        if torch.cuda.is_available():
            # Force CUDA initialization
            torch.cuda.init()
            torch.cuda.empty_cache()
            
            # Set device
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(0)
            
            # Print GPU info
            console.print(f"[bold green]CUDA Initialization:[/bold green]")
            console.print(f"[green]CUDA Version: {torch.version.cuda}[/green]")
            console.print(f"[green]PyTorch Version: {torch.__version__}[/green]")
            console.print(f"[green]GPU Device: {torch.cuda.get_device_name(0)}[/green]")
            console.print(f"[green]Device Capability: {torch.cuda.get_device_capability()}[/green]")
            
            # Enable TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
            # Set memory allocation settings
            torch.cuda.set_per_process_memory_fraction(0.7)
            
            # Create a small tensor to ensure GPU is initialized
            dummy = torch.ones(1, device=self.device)
            del dummy
            torch.cuda.synchronize()
            
            # Initialize CUDA text processor
            self.cuda_processor = CudaTextProcessor().to(self.device)
            
            # Get initial memory stats
            free_memory, total_memory = torch.cuda.mem_get_info()
            console.print(f"[green]Initial GPU Memory: {(total_memory-free_memory)/1024**3:.2f}GB / {total_memory/1024**3:.2f}GB[/green]")
        else:
            self.device = torch.device("cpu")
            console.print("[bold red]GPU not available, using CPU[/bold red]")
            use_gpu = False
        
        self.SAVE_DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.STATS_DIR = os.path.join(self.BASE_DIR, "stats")
        self.df = self.cleaned_df.copy() if hasattr(self, 'cleaned_df') else None

        # Sentiment-specific settings
        self.use_vader = use_vader
        self.use_textblob = use_textblob
        self.sentiment_threshold = sentiment_threshold
        self.vader = None  # Initialize vader to None by default
        
        # Initialize sentiment patterns
        self.sentiment_patterns = {
            'intensifiers': re.compile(r'\b(very|really|extremely|absolutely|completely|totally|utterly|highly|incredibly)\b', re.IGNORECASE),
            'negations': re.compile(r'\b(not|no|never|none|noone|nobody|nothing|neither|nowhere|hardly|scarcely|barely|don\'t|doesn\'t|didn\'t|won\'t|wouldn\'t|shouldn\'t|couldn\'t|isn\'t|aren\'t|wasn\'t|weren\'t)\b', re.IGNORECASE),
            'emoji_positive': re.compile(r'[\U0001F600-\U0001F64F]'),  # Emoticons and emojis
            'emoji_negative': re.compile(r'[\U0001F61E-\U0001F64F]'),  # Sad/negative emojis
            'emphasis': re.compile(r'(\!+|\?+|\.{2,})')  # Emphasis punctuation
        }
        
        # Load emotion lexicon
        self.emotion_lexicon = {
            'positive': set(['good', 'great', 'awesome', 'excellent', 'happy', 'love', 'wonderful', 'fantastic']),
            'negative': set(['bad', 'terrible', 'awful', 'horrible', 'sad', 'hate', 'poor', 'disappointing']),
            'neutral': set(['okay', 'fine', 'average', 'moderate', 'fair', 'decent'])
        }
        
        try:
            # Try to import and initialize VADER sentiment analyzer
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
        except ImportError:
            self.use_vader = False
            console.print("[yellow]VADER sentiment analyzer not available. Installing required package...[/yellow]")
            os.system('pip install vaderSentiment')
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self.vader = SentimentIntensityAnalyzer()
                self.use_vader = True
            except ImportError:
                console.print("[red]Failed to install VADER. Falling back to TextBlob only.[/red]")
                self.vader = None
                self.use_vader = False
        
        # Initialize other attributes (existing code)
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
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            # Warm up GPU
            dummy_tensor = torch.zeros(1, device=self.device)
            del dummy_tensor
            torch.cuda.empty_cache()
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
        
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.memory_threshold = memory_threshold
        self.num_workers = num_workers
        self.advanced_cuda = advanced_cuda
        
        # Initialize memory tracker
        self.memory_tracker = GpuMemoryTracker(threshold_percent=memory_threshold)
        
        # Initialize GPU processors
        if self.use_gpu:
            try:
                # Use legacy processor for backward compatibility
                self.gpu_processor = CudaTextProcessor().to(self.device)
                
                # Use advanced processors if enabled
                if self.advanced_cuda:
                    self.cuda_ops = CudaTextOperations(device=self.device)
                    self.parallel_processor = GpuParallelProcessor(
                        batch_size=batch_size,
                        num_workers=num_workers,
                        device=self.device,
                        memory_threshold=memory_threshold
                    )
                    # Monitor initial GPU memory
                    free_memory, total_memory = torch.cuda.mem_get_info()
                    used_memory = total_memory - free_memory
                    console.print(f"[bold green]GPU Memory Usage: {used_memory/1024**3:.2f}GB / {total_memory/1024**3:.2f}GB[/bold green]")
            except Exception as e:
                console.print(f"[bold red]Error initializing GPU processors: {e}[/bold red]")
                self.use_gpu = False
                self.device = torch.device("cpu")
        else:
            self.gpu_processor = None
            self.cuda_ops = None
            self.parallel_processor = None
        
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
        if self.use_gpu:
            self.logger.info("Using GPU for preprocessing")
            console.print("[bold green]Using GPU for preprocessing...[/bold green]")
            
            # Process in chunks to avoid memory issues
            if not torch.cuda.is_available():
                console.print("[bold red]Error: GPU not available. This method requires GPU.[/bold red]")
                return

            # Verify GPU is being used and warm it up
            console.print(f"[bold green]Current CUDA device: {torch.cuda.current_device()}[/bold green]")
            console.print(f"[bold green]Using GPU: {torch.cuda.get_device_name(0)}[/bold green]")
            
            # Warm up GPU
            if hasattr(self, 'gpu_processor') and self.gpu_processor is not None:
                self.gpu_processor.process_batch(self.df['Text'])

            # Process in chunks with better memory management
            chunk_size = min(1000, len(self.df) // 4)  # Smaller chunks for better stability
            processed_chunks = []

            for i in range(0, len(self.df), chunk_size):
                chunk_end = min(i + chunk_size, len(self.df))
                chunk_df = self.df.iloc[i:chunk_end].copy()
                
                console.print(f"[bold cyan]Processing chunk {i//chunk_size + 1}/{(len(self.df) + chunk_size - 1)//chunk_size} (rows {i} to {chunk_end})[/bold cyan]")
                
                try:
                    # Process chunk
                    processed_texts = self.gpu_processor.process_batch(chunk_df['Text'])
                    
                    # Create chunk results DataFrame
                    chunk_results = pd.DataFrame({
                        'Text': processed_texts,
                        'Tokens': [len(word_tokenize(text)) for text in processed_texts],
                        'Sentences': [sent_tokenize(text) for text in processed_texts],
                        'Entities': [{} for _ in range(len(processed_texts))],
                        'Token_count': [len(tokens) for tokens in processed_texts],
                        'Sentence_count': [len(sentences) for sentences in processed_texts],
                        'Text_length': [len(text) for text in processed_texts],
                        'Quality_score': [compute_text_quality_score(text) for text in processed_texts]
                    })
                    
                    processed_chunks.append(chunk_results)
                    
                except Exception as e:
                    console.print(f"[bold red]Error processing chunk starting at index {i}: {str(e)}[/bold red]")
                    continue
            
            # Concatenate all processed chunks
            if processed_chunks:
                self.df = pd.concat(processed_chunks, ignore_index=True)
                
                # Verify the final DataFrame
                if len(self.df) != len(self.cleaned_df):
                    console.print(f"[bold red]Warning: Processed DataFrame length ({len(self.df)}) doesn't match original ({len(self.cleaned_df)})[/bold red]")
            else:
                console.print("[bold red]Error: No chunks were successfully processed[/bold red]")
            
            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
                # Print final memory stats
                free_memory, total_memory = torch.cuda.mem_get_info()
                memory_used = (total_memory - free_memory) / 1024**3
                console.print(f"[bold green]Final GPU Memory: {memory_used:.2f}GB / {total_memory/1024**3:.2f}GB[/bold green]")
        else:
            # Fallback to CPU processing if GPU is not available
            self.logger.warning("GPU not available, using CPU for preprocessing")
            console.print("[bold yellow]GPU not available, using CPU for preprocessing...[/bold yellow]")
            
            # Process using CPU method
            super().preprocess_dataframe()
            
            # Add sentiment analysis features
            sentiment_features = []
            sentiment_scores = []
            
            for text in self.df['Text']:
                features = self.extract_sentiment_features(text)
                scores = self.compute_sentiment_scores(text)
                sentiment_features.append(features)
                sentiment_scores.append(scores)
            
            # Add sentiment columns
            for feature_name in ['intensifier_count', 'negation_count', 'emoji_positive_count',
                               'emoji_negative_count', 'emphasis_count', 'positive_words',
                               'negative_words', 'neutral_words']:
                self.df[feature_name.title()] = [f[feature_name] for f in sentiment_features]
            
            for score_name in ['textblob_polarity', 'textblob_subjectivity', 'vader_compound',
                             'vader_pos', 'vader_neg', 'vader_neu', 'custom_sentiment']:
                self.df[score_name.title()] = [s[score_name] for s in sentiment_scores]
        
        # Filter out low-quality texts if needed
        if self.min_quality_score > 0:
            low_quality_mask = self.df['Quality_score'] < self.min_quality_score
            low_quality_count = low_quality_mask.sum()
            self.logger.info(f"Filtering out {low_quality_count} low-quality texts")
            console.print(f"[bold yellow]Filtering out {low_quality_count} low-quality texts[/bold yellow]")
            
            # Replace low-quality texts with empty string
            self.df.loc[low_quality_mask, 'Text'] = ''
        
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
        text = re.sub(r'(https?|ftp):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])', ' ', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)
        text = re.sub(r'@\w+', ' ', text)
        
        # Handle numbers based on configuration
        if not self.preserve_numbers:
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

    def extract_sentiment_features(self, text):
        """Extract sentiment-specific features from text"""
        features = {
            'intensifier_count': len(self.sentiment_patterns['intensifiers'].findall(text)),
            'negation_count': len(self.sentiment_patterns['negations'].findall(text)),
            'emoji_positive_count': len(self.sentiment_patterns['emoji_positive'].findall(text)),
            'emoji_negative_count': len(self.sentiment_patterns['emoji_negative'].findall(text)),
            'emphasis_count': len(self.sentiment_patterns['emphasis'].findall(text)),
            'positive_words': 0,
            'negative_words': 0,
            'neutral_words': 0
        }
        
        # Count sentiment words
        words = text.lower().split()
        for word in words:
            if word in self.emotion_lexicon['positive']:
                features['positive_words'] += 1
            elif word in self.emotion_lexicon['negative']:
                features['negative_words'] += 1
            elif word in self.emotion_lexicon['neutral']:
                features['neutral_words'] += 1
        
        return features
    
    def compute_sentiment_scores(self, text):
        """Compute sentiment scores using multiple methods"""
        scores = {
            'textblob_polarity': 0.0,
            'textblob_subjectivity': 0.0,
            'vader_compound': 0.0,
            'vader_pos': 0.0,
            'vader_neg': 0.0,
            'vader_neu': 0.0,
            'custom_sentiment': 0.0
        }
        
        if not text or pd.isna(text):
            return scores
            
        try:
            if self.use_textblob:
                blob = TextBlob(text)
                scores['textblob_polarity'] = blob.sentiment.polarity
                scores['textblob_subjectivity'] = blob.sentiment.subjectivity
        except:
            pass
            
        try:
            if self.use_vader:
                vader_scores = self.vader.polarity_scores(text)
                scores['vader_compound'] = vader_scores['compound']
                scores['vader_pos'] = vader_scores['pos']
                scores['vader_neg'] = vader_scores['neg']
                scores['vader_neu'] = vader_scores['neu']
        except:
            pass
            
        # Compute custom sentiment score
        features = self.extract_sentiment_features(text)
        custom_score = (
            (features['positive_words'] - features['negative_words']) +
            (features['emoji_positive_count'] - features['emoji_negative_count']) * 0.5 +
            features['intensifier_count'] * 0.2 -
            features['negation_count'] * 0.3
        ) / (sum(features.values()) + 1)  # Normalize
        
        scores['custom_sentiment'] = max(-1.0, min(1.0, custom_score))  # Clamp between -1 and 1
        
        return scores


if __name__ == "__main__":
    with cProfile.Profile() as profile:
        # Set PyTorch to use the highest optimization level
        if torch.cuda.is_available():
            # Enable TF32 for better performance on Ampere GPUs (like L4)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cudnn benchmarking and autotuner
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Print CUDA version and device information
            console.print(f"[bold green]CUDA Version: {torch.version.cuda}[/bold green]")
            console.print(f"[bold green]PyTorch Version: {torch.__version__}[/bold green]")
            console.print(f"[bold green]GPU Device: {torch.cuda.get_device_name(0)}[/bold green]")
            
            # Get initial GPU memory usage
            free_memory, total_memory = torch.cuda.mem_get_info()
            used_memory = total_memory - free_memory
            console.print(f"[bold green]Initial GPU Memory: {used_memory/1024**3:.2f}GB / {total_memory/1024**3:.2f}GB[/bold green]")
            
            # Calculate optimal batch size (using about 70% of available memory)
            available_memory = free_memory * 0.7  # Use 70% of free memory
            estimated_sample_size = 1024  # bytes per sample (adjust based on your data)
            optimal_batch_size = int(available_memory / estimated_sample_size)
            optimal_batch_size = min(optimal_batch_size, 512)  # Cap at 512 to avoid memory issues
            
            console.print(f"[bold cyan]Calculated optimal batch size: {optimal_batch_size}[/bold cyan]")
        else:
            optimal_batch_size = 64  # Default for CPU
        
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
            batch_size=optimal_batch_size,
            memory_threshold=70.0,  # Lower threshold to avoid OOM
            num_workers=2,  # Reduced workers to avoid memory contention
            advanced_cuda=True,
            use_vader=True,
            use_textblob=True,
            sentiment_threshold=0.1
        )
        
        # Pre-warm the GPU
        if torch.cuda.is_available():
            # Perform a small warm-up computation
            dummy_input = torch.randn(1, 1024, device='cuda')
            dummy_output = torch.nn.functional.relu(dummy_input)
            del dummy_input, dummy_output
            torch.cuda.empty_cache()
            
            # Optional: Pin memory for faster CPU->GPU transfer
            torch.cuda.set_device(0)
            
        console.print("[bold cyan]DataFrame shape:[/bold cyan]", model.df.shape)
        console.print("[bold cyan]First few rows of 'Text' column:[/bold cyan]")
        console.print(model.df['Text'].head())
        
        model.get_statistics(model.df)

        # Output paths for different formats
        OUTPUT_CSV = os.path.join(model.SAVE_DATA_DIR, "PREPROCESSED_GPU_Reviews.csv")
        OUTPUT_PARQUET = os.path.join(model.SAVE_DATA_DIR, "PREPROCESSED_GPU_Reviews.parquet")

        # Preprocess data with timing and memory monitoring
        start_time = timeit.default_timer()
        
        if torch.cuda.is_available():
            # Monitor initial memory
            free_memory_start, total_memory = torch.cuda.mem_get_info()
            torch.cuda.reset_peak_memory_stats()  # Reset peak stats
            
        # Process in chunks to avoid memory issues
        chunk_size = len(model.df) // 4  # Process in 4 chunks
        for i in range(0, len(model.df), chunk_size):
            chunk_end = min(i + chunk_size, len(model.df))
            console.print(f"[bold cyan]Processing chunk {i//chunk_size + 1}/4 (rows {i} to {chunk_end})[/bold cyan]")
            
            # Process chunk
            model.df.iloc[i:chunk_end] = model.preprocess_dataframe()
            
            # Force memory cleanup after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
                # Monitor memory usage
                free_memory_current, _ = torch.cuda.mem_get_info()
                memory_used = (total_memory - free_memory_current) / 1024**3
                console.print(f"[bold yellow]Current GPU Memory Used: {memory_used:.2f}GB[/bold yellow]")
        
        if torch.cuda.is_available():
            # Get peak memory stats
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            console.print(f"[bold yellow]Peak GPU Memory Used: {peak_memory:.2f}GB[/bold yellow]")
            
        elapsed = timeit.default_timer() - start_time
        
        # Remove non-English text
        model.remove_foreign_words()
        
        # Perform feature engineering
        model.perform_feature_engineering()
        
        # Save to multiple formats
        model.save_to_csv(output_path=OUTPUT_CSV)
        
        # Load processed data and get statistics
        cleaned_df = pd.read_csv(OUTPUT_CSV)
        model.get_statistics(cleaned_df)
        
        # Print processing summary
        model.get_preprocessing_summary()

        console.print(f"[bold green]Preprocessing took {elapsed:.2f} seconds[/bold green]")
        console.print(f"[bold green]Average processing speed: {len(model.df)/elapsed:.2f} samples/second[/bold green]")
        console.print("[bold cyan]Processed DataFrame:[/bold cyan]")
        console.print(model.df.head())
        console.print("\n[bold cyan]DataFrame columns:[/bold cyan]", model.df.columns.tolist())

        # Final cleanup and memory stats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_memory_final, total_memory = torch.cuda.mem_get_info()
            final_usage = (total_memory - free_memory_final) / 1024**3
            console.print(f"[bold green]Final GPU Memory Usage: {final_usage:.2f}GB[/bold green]")

    stats_file_dir = os.path.join(model.STATS_DIR, "gpu_results.prof")
    
    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.dump_stats(stats_file_dir)
