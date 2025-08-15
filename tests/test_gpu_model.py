import sys
import os
import timeit
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import preprocessing models
from src.data.preprocessing_model import PreprocessingModel
from src.data.preprocessing_model_gpu import PreprocessingModelGPU

console = Console()


def test_preprocessing_speed(sample_size=1000, batch_size=32):
    """
    Compare the speed of the original and GPU-optimized preprocessing models
    
    Args:
        sample_size (int): Number of texts to process
        batch_size (int): Batch size for GPU processing
    """
    console.print("[bold cyan]Testing preprocessing speed...[/bold cyan]")
    
    # Load data
    console.print("[yellow]Loading data...[/yellow]")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(base_dir, "data", "Reviews.csv")
    
    try:
        # Load a sample of the data
        df = pd.read_csv(data_path)
        sample_df = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
        
        console.print(f"[green]Loaded {len(sample_df)} samples for testing[/green]")
    except Exception as e:
        console.print(f"[bold red]Error loading data: {e}[/bold red]")
        return
    
    # Initialize models
    try:
        # Standard CPU model
        cpu_model = PreprocessingModel(
            use_lemmatization=True,
            use_stemming=False,
            preserve_negation=True, 
            preserve_named_entities=True,
            preserve_numbers=True,
            correct_spelling=True,
            advanced_tokenization=True,
            use_spacy=True,
            remove_duplicates=True,
            min_quality_score=0.6
        )
        cpu_model.df = sample_df.copy()
        
        # GPU model
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
            batch_size=batch_size
        )
        gpu_model.df = sample_df.copy()
        
        console.print("[green]Models initialized successfully[/green]")
    except Exception as e:
        console.print(f"[bold red]Error initializing models: {e}[/bold red]")
        return
    
    # Time CPU model
    console.print("[yellow]Testing CPU model...[/yellow]")
    cpu_start = timeit.default_timer()
    try:
        cpu_model.preprocess_dataframe()
        cpu_elapsed = timeit.default_timer() - cpu_start
        console.print(f"[green]CPU model finished in {cpu_elapsed:.2f} seconds[/green]")
    except Exception as e:
        console.print(f"[bold red]CPU model error: {e}[/bold red]")
        cpu_elapsed = float('inf')
    
    # Time GPU model
    console.print("[yellow]Testing GPU model...[/yellow]")
    gpu_start = timeit.default_timer()
    try:
        gpu_model.preprocess_dataframe()
        gpu_elapsed = timeit.default_timer() - gpu_start
        console.print(f"[green]GPU model finished in {gpu_elapsed:.2f} seconds[/green]")
    except Exception as e:
        console.print(f"[bold red]GPU model error: {e}[/bold red]")
        gpu_elapsed = float('inf')
    
    # Compare results
    if cpu_elapsed != float('inf') and gpu_elapsed != float('inf'):
        speedup = cpu_elapsed / gpu_elapsed if gpu_elapsed > 0 else float('inf')
        
        results = Table(title="Performance Comparison", show_header=True, header_style="bold magenta")
        results.add_column("Model", style="cyan")
        results.add_column("Time (s)", style="green")
        results.add_column("Samples/s", style="yellow")
        results.add_column("Speedup", style="red")
        
        results.add_row(
            "CPU Model", 
            f"{cpu_elapsed:.2f}", 
            f"{sample_size/cpu_elapsed:.2f}",
            "1.00x"
        )
        results.add_row(
            "GPU Model", 
            f"{gpu_elapsed:.2f}", 
            f"{sample_size/gpu_elapsed:.2f}",
            f"{speedup:.2f}x"
        )
        
        console.print(results)
        
        # Check for any differences in the results
        try:
            cpu_df = cpu_model.df.copy()
            gpu_df = gpu_model.df.copy()
            
            # Count non-empty texts
            cpu_texts = (cpu_df['Text'] != '').sum()
            gpu_texts = (gpu_df['Text'] != '').sum()
            
            if abs(cpu_texts - gpu_texts) / max(cpu_texts, gpu_texts) > 0.1:  # More than 10% difference
                console.print(f"[bold yellow]Warning: Significant difference in results. CPU: {cpu_texts} non-empty texts, GPU: {gpu_texts} non-empty texts[/bold yellow]")
            else:
                console.print(f"[green]Results are comparable: CPU: {cpu_texts} non-empty texts, GPU: {gpu_texts} non-empty texts[/green]")
        except Exception as e:
            console.print(f"[bold red]Error comparing results: {e}[/bold red]")
    
    # Check GPU memory usage if available
    if torch.cuda.is_available():
        try:
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**2  # MB
            
            console.print(f"[bold cyan]GPU Memory Usage:[/bold cyan]")
            console.print(f"  Allocated: {mem_allocated:.2f} MB")
            console.print(f"  Reserved:  {mem_reserved:.2f} MB")
            
            # Free memory
            torch.cuda.empty_cache()
        except Exception as e:
            console.print(f"[bold red]Error checking GPU memory: {e}[/bold red]")


def test_length_mismatch_fix(rows=1000):
    """
    Test if the GPU-optimized model fixes the length mismatch error
    
    Args:
        rows (int): Number of rows to process
    """
    console.print("[bold cyan]Testing length mismatch fix...[/bold cyan]")
    
    # Load data
    console.print("[yellow]Loading data...[/yellow]")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
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
        
        # Run preprocessing
        console.print("[yellow]Processing with GPU model...[/yellow]")
        gpu_model.preprocess_dataframe()
        
        # Check for length mismatches
        original_length = len(sample_df)
        result_length = len(gpu_model.df)
        
        if original_length == result_length:
            console.print(f"[bold green]Success! No length mismatch. Original: {original_length}, Result: {result_length}[/bold green]")
        else:
            console.print(f"[bold red]Length mismatch still exists. Original: {original_length}, Result: {result_length}[/bold red]")
        
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
    console.print("[bold]===== GPU Preprocessing Model Tests =====[/bold]")
    
    # Check if GPU is available
    if torch.cuda.is_available():
        console.print(f"[bold green]GPU available: {torch.cuda.get_device_name(0)}[/bold green]")
    else:
        console.print("[bold yellow]No GPU available. Will run in CPU mode.[/bold yellow]")
    
    # Run tests
    test_length_mismatch_fix(rows=500)
    test_preprocessing_speed(sample_size=500, batch_size=32)
