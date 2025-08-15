import sys
import os
import pandas as pd
from rich.console import Console

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import preprocessing models
from src.data.preprocessing_model_gpu import PreprocessingModelGPU

console = Console()

def test_length_mismatch_fix(rows=500):
    """Test if the GPU-optimized model fixes the length mismatch error"""
    console.print("[bold cyan]Testing length mismatch fix...[/bold cyan]")
    
    # Load data
    console.print("[yellow]Loading data...[/yellow]")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    data_path = os.path.join(base_dir, "data", "Reviews.csv")
    
    try:
        # Load a sample of the data
        df = pd.read_csv(data_path)
        sample_df = df.sample(n=rows, random_state=42) if len(df) > rows else df
        
        console.print(f"[green]Loaded {len(sample_df)} samples for testing[/green]")
    except Exception as e:
        console.print(f"[bold red]Error loading data: {e}[/bold red]")
        return
    
    # Test GPU model
    try:
        gpu_model = PreprocessingModelGPU(
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
            use_gpu=True,
            batch_size=32
        )
        gpu_model.df = sample_df.copy()
        
        # Print original dataframe info
        console.print(f"[yellow]Original DataFrame shape: {gpu_model.df.shape}[/yellow]")
        
        # Run preprocessing
        console.print("[yellow]Processing with GPU model...[/yellow]")
        gpu_model.preprocess_dataframe()
        
        # Check for length mismatches
        original_length = len(sample_df)
        result_length = len(gpu_model.df)
        
        console.print(f"[yellow]Result DataFrame shape: {gpu_model.df.shape}[/yellow]")
        
        if original_length == result_length:
            console.print(f"[bold green]Success! No length mismatch. Original: {original_length}, Result: {result_length}[/bold green]")
        else:
            console.print(f"[bold red]Length mismatch still exists. Original: {original_length}, Result: {result_length}[/bold red]")
        
        # Check for empty text counts
        empty_count = (gpu_model.df['Text'] == '').sum()
        console.print(f"[yellow]Empty text count: {empty_count}[/yellow]")
        
        # Save a small sample of results for inspection
        output_path = os.path.join(base_dir, "data", "gpu_test_results.csv")
        gpu_model.df.head(100).to_csv(output_path, index=False)
        console.print(f"[green]Saved sample results to {output_path}[/green]")
        
    except ValueError as e:
        if "Length of values" in str(e):
            console.print(f"[bold red]Length mismatch error: {e}[/bold red]")
        else:
            console.print(f"[bold red]Error: {e}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {e}[/bold red]")

if __name__ == "__main__":
    test_length_mismatch_fix(500)
