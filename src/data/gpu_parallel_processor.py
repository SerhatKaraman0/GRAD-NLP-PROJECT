"""
GPU-accelerated parallel processing module for text data.
Enables efficient batch processing with advanced parallelization.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor
import logging
from tqdm import tqdm
from rich.console import Console
from torch.utils.data import DataLoader, TensorDataset, Dataset

# Import our custom modules
from src.data.cuda_text_kernels import CudaTextOperations, batch_process_with_cuda
from src.data.gpu_memory_manager import GpuMemoryTracker, clear_gpu_memory, get_gpu_memory_status

console = Console()
logger = logging.getLogger(__name__)


class ParallelTextDataset(Dataset):
    """
    Dataset for parallel processing of text data.
    Supports both CPU and GPU operations.
    """
    
    def __init__(self, texts, metadata=None, text_column='Text'):
        """
        Initialize dataset.
        
        Args:
            texts (pd.DataFrame or List): Input texts or dataframe
            metadata (Dict): Additional metadata
            text_column (str): Column name if texts is a dataframe
        """
        if isinstance(texts, pd.DataFrame):
            self.texts = texts[text_column].tolist()
            self.df = texts
        else:
            self.texts = texts
            self.df = None
        
        self.metadata = metadata or {}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]
    
    def get_batch(self, indices):
        """Get a batch of texts by indices"""
        return [self.texts[i] for i in indices]
    
    def get_metadata_for_batch(self, indices):
        """Get metadata for a batch by indices"""
        return {k: [v[i] for i in indices] for k, v in self.metadata.items()}


class GpuParallelProcessor:
    """
    Parallel text processor using GPU acceleration.
    """
    
    def __init__(self, 
                 batch_size=32, 
                 num_workers=4,
                 device=None,
                 memory_threshold=80.0):
        """
        Initialize processor.
        
        Args:
            batch_size (int): Processing batch size
            num_workers (int): Number of CPU workers for data loading
            device (torch.device): Device to use
            memory_threshold (float): GPU memory threshold percentage
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_tracker = GpuMemoryTracker(threshold_percent=memory_threshold)
        self.cuda_ops = CudaTextOperations(device=self.device)
    
    def create_dataloader(self, dataset, shuffle=False):
        """Create a DataLoader for the dataset"""
        return DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def process_dataframe(self, 
                          df: pd.DataFrame,
                          text_column: str,
                          operations: Dict[str, Any]) -> pd.DataFrame:
        """
        Process a dataframe with GPU acceleration.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Column containing text
            operations (Dict[str, Any]): Operations to apply
            
        Returns:
            pd.DataFrame: Processed dataframe
        """
        if df is None or df.empty or text_column not in df.columns:
            logger.error("Invalid dataframe or text column")
            return df
        
        # Create dataset and dataloader
        dataset = ParallelTextDataset(df, text_column=text_column)
        dataloader = self.create_dataloader(dataset)
        
        # Start memory tracking
        self.memory_tracker.start_tracking()
        
        # Process in batches
        all_results = {}
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Check and clear memory if needed
            memory_percent, cleared = self.memory_tracker.check_memory()
            if cleared:
                logger.info(f"Cleared GPU memory at batch {batch_idx}. Memory was at {memory_percent:.2f}%")
            
            # Process batch with CUDA operations
            batch_results = batch_process_with_cuda(
                batch,
                operations,
                batch_size=self.batch_size
            )
            
            # Merge batch results
            for op_name, res in batch_results.items():
                if op_name not in all_results:
                    all_results[op_name] = []
                all_results[op_name].extend(res)
        
        # Stop memory tracking
        memory_stats = self.memory_tracker.stop_tracking()
        logger.info(f"GPU memory usage - Peak: {memory_stats['peak_mb']:.2f} MB, "
                   f"Final: {memory_stats['final_mb']:.2f} MB")
        
        # Create result dataframe
        result_df = df.copy()
        
        # Add processed columns
        for op_name, res in all_results.items():
            # Ensure result length matches dataframe length
            if len(res) != len(df):
                logger.warning(f"Length mismatch for operation {op_name}: {len(res)} vs {len(df)}")
                
                # Fix length mismatch
                if len(res) < len(df):
                    # Pad with empty values
                    if isinstance(res[0], list):
                        res.extend([[] for _ in range(len(df) - len(res))])
                    elif isinstance(res[0], dict):
                        res.extend([{} for _ in range(len(df) - len(res))])
                    elif isinstance(res[0], (int, float)):
                        res.extend([0 for _ in range(len(df) - len(res))])
                    else:
                        res.extend([''] * (len(df) - len(res)))
                else:
                    # Truncate
                    res = res[:len(df)]
            
            # Add to dataframe
            result_df[f"{op_name}_{text_column}"] = res
        
        return result_df
    
    def parallel_apply(self, 
                       texts: List[str], 
                       fn: Callable,
                       **fn_kwargs) -> List[Any]:
        """
        Apply a function to texts in parallel batches.
        
        Args:
            texts (List[str]): Input texts
            fn (Callable): Function to apply
            **fn_kwargs: Keyword arguments for fn
            
        Returns:
            List[Any]: Results of function application
        """
        # Create dataset and dataloader
        dataset = ParallelTextDataset(texts)
        dataloader = self.create_dataloader(dataset)
        
        # Process in batches
        all_results = []
        
        for batch in tqdm(dataloader, desc="Parallel processing"):
            # Apply function to batch
            batch_results = fn(batch, **fn_kwargs)
            all_results.extend(batch_results)
        
        return all_results
    
    @staticmethod
    def gpu_available_check():
        """Check if GPU is available and print device info"""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            console.print(f"[bold green]GPU available: {device_name} with {memory:.2f} GB memory[/bold green]")
            return True
        else:
            console.print("[bold yellow]GPU not available, using CPU[/bold yellow]")
            return False
