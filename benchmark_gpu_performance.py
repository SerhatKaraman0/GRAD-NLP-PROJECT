"""
Benchmarking tool for GPU preprocessing performance.
Tests and compares performance of different preprocessing configurations.
"""
import sys
import os
import timeit
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
import json
from datetime import datetime
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import preprocessing models
from src.data.preprocessing_model import PreprocessingModel
from src.data.preprocessing_model_gpu import PreprocessingModelGPU
from src.data.gpu_memory_manager import GpuMemoryTracker, get_gpu_memory_status

console = Console()
logger = logging.getLogger(__name__)


def setup_logging():
    """Set up logging configuration"""
    log_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs')))
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def run_benchmark(sample_sizes=None, batch_sizes=None, num_runs=3):
    """
    Run comprehensive benchmarking of GPU vs CPU preprocessing.
    
    Args:
        sample_sizes (List[int]): Sample sizes to test
        batch_sizes (List[int]): Batch sizes to test
        num_runs (int): Number of runs for each configuration
    """
    setup_logging()
    console.print("[bold cyan]Starting GPU/CPU Preprocessing Benchmark[/bold cyan]")
    
    # Default test configurations
    if sample_sizes is None:
        sample_sizes = [100, 500, 1000, 5000]
    
    if batch_sizes is None:
        batch_sizes = [16, 32, 64, 128]
    
    # Load data
    console.print("[yellow]Loading data...[/yellow]")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(base_dir, "data", "Reviews.csv")
    
    try:
        # Load the data
        df = pd.read_csv(data_path)
        console.print(f"[green]Loaded dataset with {len(df)} rows[/green]")
    except Exception as e:
        console.print(f"[bold red]Error loading data: {e}[/bold red]")
        return
    
    # Initialize results containers
    results = {
        "configurations": [],
        "cpu_times": [],
        "gpu_times": [],
        "speedup_factors": [],
        "memory_usage": [],
        "sample_size": [],
        "batch_size": []
    }
    
    # Benchmark different configurations
    for sample_size in sample_sizes:
        # Get sample
        if len(df) > sample_size:
            sample_df = df.sample(n=sample_size, random_state=42)
        else:
            sample_df = df
            sample_size = len(df)
        
        console.print(f"\n[bold cyan]Testing with sample size: {sample_size}[/bold cyan]")
        
        # Test CPU model first (batch size doesn't matter)
        cpu_times = []
        for run in range(num_runs):
            console.print(f"[yellow]CPU Run {run+1}/{num_runs}...[/yellow]")
            
            # Initialize CPU model
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
            
            # Time CPU model
            cpu_start = timeit.default_timer()
            cpu_model.preprocess_dataframe()
            cpu_elapsed = timeit.default_timer() - cpu_start
            
            cpu_times.append(cpu_elapsed)
            console.print(f"[green]CPU finished in {cpu_elapsed:.2f} seconds[/green]")
        
        avg_cpu_time = sum(cpu_times) / len(cpu_times)
        console.print(f"[bold green]Average CPU time: {avg_cpu_time:.2f} seconds[/bold green]")
        
        # Test GPU model with different batch sizes
        for batch_size in batch_sizes:
            console.print(f"\n[bold cyan]Testing with batch size: {batch_size}[/bold cyan]")
            
            gpu_times = []
            peak_memory = []
            
            for run in range(num_runs):
                console.print(f"[yellow]GPU Run {run+1}/{num_runs} with batch size {batch_size}...[/yellow]")
                
                # Clear GPU memory between runs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Initialize memory tracker
                memory_tracker = GpuMemoryTracker()
                memory_tracker.start_tracking()
                
                # Initialize GPU model
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
                
                # Time GPU model
                gpu_start = timeit.default_timer()
                gpu_model.preprocess_dataframe()
                gpu_elapsed = timeit.default_timer() - gpu_start
                
                # Get memory stats
                memory_stats = memory_tracker.stop_tracking()
                peak_memory.append(memory_stats["peak_mb"])
                
                gpu_times.append(gpu_elapsed)
                console.print(f"[green]GPU finished in {gpu_elapsed:.2f} seconds. "
                             f"Peak memory: {memory_stats['peak_mb']:.2f} MB[/green]")
            
            avg_gpu_time = sum(gpu_times) / len(gpu_times)
            avg_memory = sum(peak_memory) / len(peak_memory)
            speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 0
            
            console.print(f"[bold green]Average GPU time: {avg_gpu_time:.2f} seconds[/bold green]")
            console.print(f"[bold green]Average memory usage: {avg_memory:.2f} MB[/bold green]")
            console.print(f"[bold green]Speedup factor: {speedup:.2f}x[/bold green]")
            
            # Add to results
            results["configurations"].append(f"Sample: {sample_size}, Batch: {batch_size}")
            results["cpu_times"].append(avg_cpu_time)
            results["gpu_times"].append(avg_gpu_time)
            results["speedup_factors"].append(speedup)
            results["memory_usage"].append(avg_memory)
            results["sample_size"].append(sample_size)
            results["batch_size"].append(batch_size)
    
    # Create summary table
    table = Table(title="Preprocessing Benchmark Results")
    table.add_column("Configuration", justify="left")
    table.add_column("CPU Time (s)", justify="right")
    table.add_column("GPU Time (s)", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Memory (MB)", justify="right")
    
    for i in range(len(results["configurations"])):
        table.add_row(
            results["configurations"][i],
            f"{results['cpu_times'][i]:.2f}",
            f"{results['gpu_times'][i]:.2f}",
            f"{results['speedup_factors'][i]:.2f}x",
            f"{results['memory_usage'][i]:.2f}"
        )
    
    console.print(table)
    
    # Save results to CSV
    results_df = pd.DataFrame({
        "Sample Size": results["sample_size"],
        "Batch Size": results["batch_size"],
        "CPU Time (s)": results["cpu_times"],
        "GPU Time (s)": results["gpu_times"],
        "Speedup Factor": results["speedup_factors"],
        "Memory Usage (MB)": results["memory_usage"]
    })
    
    output_path = os.path.join(base_dir, "data", "gpu_benchmark_results.csv")
    results_df.to_csv(output_path, index=False)
    console.print(f"[green]Saved benchmark results to {output_path}[/green]")
    
    # Create visualization
    try:
        plot_benchmark_results(results)
    except Exception as e:
        console.print(f"[yellow]Error creating visualization: {e}[/yellow]")


def plot_benchmark_results(results):
    """Create visualization of benchmark results"""
    # Convert to numpy arrays for easier manipulation
    sample_sizes = np.array(results["sample_size"])
    batch_sizes = np.array(results["batch_size"])
    unique_samples = np.unique(sample_sizes)
    unique_batches = np.unique(batch_sizes)
    
    # Create speedup comparison plot
    plt.figure(figsize=(12, 8))
    
    for sample_size in unique_samples:
        indices = sample_sizes == sample_size
        plt.plot(
            batch_sizes[indices], 
            results["speedup_factors"][indices], 
            'o-',
            label=f"Sample size: {sample_size}"
        )
    
    plt.title("GPU Speedup Factor by Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Speedup Factor (CPU time / GPU time)")
    plt.grid(True)
    plt.legend()
    
    # Save plot
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    plot_path = os.path.join(base_dir, "data", "metrics", "gpu_speedup_comparison.png")
    plt.savefig(plot_path)
    
    # Create memory usage plot
    plt.figure(figsize=(12, 8))
    
    for sample_size in unique_samples:
        indices = sample_sizes == sample_size
        plt.plot(
            batch_sizes[indices], 
            results["memory_usage"][indices], 
            'o-',
            label=f"Sample size: {sample_size}"
        )
    
    plt.title("GPU Memory Usage by Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Memory Usage (MB)")
    plt.grid(True)
    plt.legend()
    
    # Save plot
    memory_plot_path = os.path.join(base_dir, "data", "metrics", "gpu_memory_usage.png")
    plt.savefig(memory_plot_path)
    
    console.print(f"[green]Saved visualization to {plot_path} and {memory_plot_path}[/green]")


if __name__ == "__main__":
    # Run benchmark with default settings
    run_benchmark()
