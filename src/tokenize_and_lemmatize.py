from src.nlpmodel import NlpModel
import os
import sys
import pandas as pd
import torch
import spacy
from transformers import AutoTokenizer
from rich.console import Console
from rich.progress import Progress
from typing import Dict
from multiprocessing import Pool, cpu_count, get_context

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

console = Console()

class TokenizeAndLemmatize(NlpModel):
    def __init__(self):
        super().__init__()  # Fix: Ensure proper initialization
        self.PREPROCESSED_DATA = os.path.join(self.BASE_DIR, "data", "PREPROCESSED_Reviews.csv")

        try:
            self.df = pd.read_csv(self.PREPROCESSED_DATA)
            console.print(f"[green]Successfully loaded data from {self.PREPROCESSED_DATA}[/green]")
        except FileNotFoundError:
            console.print(f"[red]Error: Could not find {self.PREPROCESSED_Reviews}[/red]")
            raise

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.nlp = spacy.load("en_core_web_sm")  # Load spaCy English model
        console.print("[green]Initialized BERT tokenizer and spaCy model[/green]")

    def lemmatize_text(self, text: str) -> str:
        """Lemmatize a single text string"""
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc])

    def lemmatize_df(self):
        """Lemmatize all texts in the DataFrame with a progress bar"""
        console.print("[yellow]Starting lemmatization process...[/yellow]")

        if "Text" not in self.df.columns:
            raise ValueError("DataFrame does not contain 'Text' column")

        # Using multiprocessing to process texts in parallel
        n_cores = min(cpu_count() or 4, 8)
        with get_context("spawn").Pool(processes=n_cores) as pool:
            lemmatized_texts = pool.map(self.lemmatize_text, self.df["Text"])

        self.df["lemmatized"] = lemmatized_texts
        console.print("[green]Lemmatization complete![/green]")

    def tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize a single text string"""
        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

    def tokenize_df(self):
        """Tokenize all texts in the DataFrame with a progress bar"""
        console.print("[yellow]Starting tokenization process...[/yellow]")

        if "lemmatized" not in self.df.columns:
            raise ValueError("DataFrame does not contain 'lemmatized' column. Run lemmatization first.")

        # Using multiprocessing to process texts in parallel
        n_cores = min(cpu_count() or 4, 8)
        with get_context("spawn").Pool(processes=n_cores) as pool:
            tokenized_texts = pool.map(self.tokenize_text, self.df["lemmatized"])

        # Prepare the tokenized DataFrame
        self.df["tokenized"] = [{
            'input_ids': tokens['input_ids'].squeeze().tolist(),
            'attention_mask': tokens['attention_mask'].squeeze().tolist()
        } for tokens in tokenized_texts]

        console.print("[green]Tokenization complete![/green]")

    def save_tokenized_data(self, output_path: str = None):
        """Save tokenized DataFrame to CSV"""
        if output_path is None:
            output_path = os.path.join(self.BASE_DIR, "data", "tokenized_data.csv")
        
        self.df.to_csv(output_path, index=False)
        console.print(f"[green]Saved tokenized data to {output_path}[/green]")

    def process_texts(self):
        """Run lemmatization and tokenization in sequence"""
        self.lemmatize_df()  # Lemmatize first
        self.tokenize_df()   # Then tokenize after lemmatization

if __name__ == "__main__":
    try:
        console.print("\n Process Started..\n")

        tokenization_model = TokenizeAndLemmatize()
        tokenization_model.process_texts()

        console.print("\n[bold cyan]Sample of processed data:[/bold cyan]")
        console.print(tokenization_model.df[["Text", "lemmatized", "tokenized"]].head())

        tokenization_model.save_tokenized_data()
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
