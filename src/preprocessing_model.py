from src.common_imports import * # noqa: F403, F405
from src.nlpmodel import NlpModel
from src.logging_config import *  # noqa: F403, F405
from utils.helper import CONTRACTIONS_DICT, SLANG_DICT  
from nltk.tokenize import word_tokenize, sent_tokenize
import gc
import itertools
from multiprocessing import get_context
from nltk.corpus import stopwords
import swifter
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

warnings.filterwarnings("ignore", category=FutureWarning)

console = Console()

class PreprocessingModel(NlpModel):
    __slots__ = ["SAVE_DATA_DIR", "STATS_DIR", "df", "patterns", "word_replacements"]

    def __init__(self):
        super().__init__()
        self.SAVE_DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.STATS_DIR = os.path.join(self.BASE_DIR, "stats")
        self.df = self.cleaned_df.copy()  

        self.patterns = {
            'html': re.compile(r"<[^>]+>"),
            'url': re.compile(r"(https?|ftp):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])"),
            'punctuation': re.compile(r"[!\'#\$%&\'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`{\|}~]"),
            'extra_punctuation': re.compile(r'(\W)\1+'),
            'whitespace': re.compile(r'\s+')
        }
        
        self.word_replacements = {**SLANG_DICT, **CONTRACTIONS_DICT}
    
    def _count_pattern_matches(self, pattern_name: str, df) -> tuple:
        self.logger.info("GETTING PATTERN MATCHES..")
        self.print_section("GETTING PATTERN MATCHES..")
        
        pattern = self.patterns[pattern_name]
        mask = df["Text"].astype(str).str.contains(pattern, regex=True, na=False)
        count = mask.sum()
        percentage = 100 * count / len(self.df)
        return count, percentage
    
    def get_statistics(self, df) -> None:
        """Efficiently calculate all statistics at once"""
        
        self.logger.info("GETTING STATISTICS..")
        self.print_section("GETTING STATISTICS..")

        table = Table(title="Pattern Statistics", show_header=True, header_style="bold magenta")
        table.add_column("Pattern", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Percentage", style="yellow")

        for pattern_name in ['html', 'url']:
            count, percentage = self._count_pattern_matches(pattern_name, df)
            table.add_row(pattern_name, str(count), f"{percentage:.2f}%")

        console.print(table)

    def _process_text(self, text):
        """Process a single text - for use with swifter"""
        text = str(text).lower()
        
        converted_text = emoji.demojize(text)
        converted_text = self.patterns['html'].sub('', converted_text)
        converted_text = self.patterns['url'].sub('', converted_text)
        converted_text = self.patterns['whitespace'].sub(' ', converted_text)
        converted_text = self.patterns['extra_punctuation'].sub(r'\1', converted_text)
        
        sentences = sent_tokenize(' '.join(converted_text.split()))
        
        converted_text = self.patterns['punctuation'].sub('', converted_text)
        
        words = converted_text.split()
        words = [self.word_replacements.get(word, word) for word in words]
        processed_text = ' '.join(words)
        
        tokens = word_tokenize(processed_text)
        
        return processed_text, tokens, sentences

    @staticmethod
    def _process_chunk(args):
        """Process a chunk of texts in parallel using numpy for loop"""
        texts, patterns, word_replacements = args

        if len(texts) == 0:
            return [], [], []

        if isinstance(texts, list):
            texts = pd.Series(texts)

        processed = np.empty(len(texts), dtype=object)
        tokenized = np.empty(len(texts), dtype=object)
        sentence_tokenized = np.empty(len(texts), dtype=object)

        for i in range(len(texts)):
            text = str(texts.iloc[i]).lower() 
            
            converted_text = emoji.demojize(text)
            converted_text = patterns['html'].sub('', converted_text)
            converted_text = patterns['url'].sub('', converted_text)
            converted_text = patterns['whitespace'].sub(' ', converted_text)
            converted_text = patterns['extra_punctuation'].sub(r'\1', converted_text)

            sentence_tokenized[i] = sent_tokenize(' '.join(converted_text.split()))
            
            converted_text = patterns['punctuation'].sub('', converted_text)
            
            words = converted_text.split()
            words = [word_replacements.get(word, word) for word in words]
            processed_text = ' '.join(words)
            
            processed[i] = processed_text
            tokenized[i] = word_tokenize(processed_text)
            
        return processed.tolist(), tokenized.tolist(), sentence_tokenized.tolist()

    def remove_foreign_words(self) -> None:
        """Remove non-English text efficiently using swifter"""
        self.logger.info("REMOVING FOREIGN WORDS STARTED..") 
        self.print_section("REMOVING FOREIGN WORDS STARTED..") 

        def detect_language(text):
            try:
                return detect(str(text)) == 'en'
            except:
                return False
        
        # Using swifter for parallelized language detection
        is_english = self.df['Text'].swifter.apply(detect_language)
        self.df.loc[~is_english, 'Text'] = ''

    def preprocess_dataframe(self) -> None:
        """Main preprocessing pipeline with swifter for parallelization"""
        self.logger.info("PREPROCESSING STARTED..")
        self.print_section("PREPROCESSING STARTED..")
        
        if self.df.empty or self.df['Text'].empty:
            self.logger.error("DataFrame or 'Text' column is empty. Aborting preprocessing.")
            return
        
        # Determine processing approach based on data size
        if len(self.df) > 10000:  # For large datasets, use chunked approach
            total_chars = self.df['Text'].str.len().sum()
            chars_per_chunk = 500000  
            n_chunks = max(1, int(total_chars / chars_per_chunk))
            
            n_cores = os.cpu_count() or 4
            n_cores = min(n_cores, 8)
            
            chunks = np.array_split(self.df['Text'], n_chunks)
            process_args = [(chunk, self.patterns, self.word_replacements) for chunk in chunks]
            
            with get_context('spawn').Pool(processes=n_cores) as pool:
                results = list(track(
                    pool.imap_unordered(self._process_chunk, process_args),
                    total=len(process_args),
                    description="Processing chunks..."
                ))
                
                processed_chunks, tokenized_chunks, sentence_tokenized_chunks = zip(*results)
                
                processed_texts = list(itertools.chain.from_iterable(processed_chunks))
                tokenized_texts = list(itertools.chain.from_iterable(tokenized_chunks))
                sentence_tokenized_texts = list(itertools.chain.from_iterable(sentence_tokenized_chunks))
                
                del processed_chunks, tokenized_chunks, sentence_tokenized_chunks
                gc.collect()
        else:
            # For smaller datasets, use swifter for parallelized apply
            console.print("[bold yellow]Using swifter for parallel processing...[/bold yellow]")
            
            # Set swifter to use a progress bar
            results = self.df['Text'].swifter.progress_bar(True).apply(self._process_text)
            
            # Unpack results
            processed_texts, tokenized_texts, sentence_tokenized_texts = zip(*results)
        
        # Create new DataFrame columns
        new_data = {
            'Text': processed_texts,
            'Tokens': tokenized_texts,
            'Sentences': sentence_tokenized_texts,
            'Token_count': [len(tokens) for tokens in tokenized_texts],
            'Sentence_count': [len(sentences) for sentences in sentence_tokenized_texts]
        }
        
        self.df = self.df[['Id', 'Score', 'Summary']].assign(**new_data)
            
        # Final cleanup
        gc.collect()

    def save_to_csv(self, output_path: str = "processed_data.csv") -> None:
        """Save the processed DataFrame to CSV"""
        self.logger.info("SAVING TO CSV STARTED..")
        self.print_section("SAVING TO CSV STARTED..")

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        console.print(f"[bold green]Saved CSV file to: {output_path}[/bold green]")

if __name__ == "__main__":
    with cProfile.Profile() as profile:
        model = PreprocessingModel()
        
        console.print("[bold cyan]DataFrame shape:[/bold cyan]", model.df.shape)
        console.print("[bold cyan]First few rows of 'Text' column:[/bold cyan]")
        console.print(model.df['Text'].head())
        
        model.get_statistics(model.df)

        OUTPUT_DIR = os.path.join(model.SAVE_DATA_DIR, "PREPROCESSED_Reviews.csv")

        start_time = timeit.default_timer()
        model.preprocess_dataframe()
        elapsed = timeit.default_timer() - start_time

        model.save_to_csv(output_path=OUTPUT_DIR)

        cleaned_df = pd.read_csv(OUTPUT_DIR)

        model.get_statistics(cleaned_df)

        console.print(f"[bold green]Preprocessing took {elapsed:.2f} seconds[/bold green]")
        console.print("[bold cyan]Processed DataFrame:[/bold cyan]")
        console.print(model.df.head())
        console.print("\n[bold cyan]DataFrame columns:[/bold cyan]", model.df.columns.tolist())

    stats_file_dir = os.path.join(model.STATS_DIR, "results.prof")
    
    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)

    results.dump_stats(stats_file_dir)