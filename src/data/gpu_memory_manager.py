"""
GPU memory management utilities for NLP processing.
Efficient memory handling for text processing operations on GPU.
"""
import torch
import gc
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GpuMemoryStatus:
    """Dataclass to hold GPU memory status information"""
    total_memory: int
    used_memory: int
    free_memory: int
    utilization_percent: float


def get_gpu_memory_status() -> Optional[GpuMemoryStatus]:
    """
    Get current GPU memory status.
    
    Returns:
        GpuMemoryStatus object or None if no GPU is available
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available - cannot get GPU memory status")
        return None
    
    try:
        # Get memory information in bytes
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        allocated_memory = torch.cuda.memory_allocated(0)
        free_memory = total_memory - reserved_memory
        
        # Calculate utilization
        utilization = (allocated_memory / total_memory) * 100
        
        return GpuMemoryStatus(
            total_memory=total_memory,
            used_memory=allocated_memory,
            free_memory=free_memory,
            utilization_percent=utilization
        )
    except Exception as e:
        logger.error(f"Error getting GPU memory status: {e}")
        return None


def clear_gpu_memory() -> bool:
    """
    Clear unused GPU memory.
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available - cannot clear GPU memory")
        return False
    
    try:
        # Empty cache to free memory
        torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        return True
    except Exception as e:
        logger.error(f"Error clearing GPU memory: {e}")
        return False


class GpuMemoryTracker:
    """
    Class to track GPU memory usage during processing
    with automatic clearance if memory exceeds threshold.
    """
    
    def __init__(self, threshold_percent: float = 80.0, auto_clear: bool = True):
        """
        Initialize memory tracker.
        
        Args:
            threshold_percent (float): Memory utilization threshold (0-100)
            auto_clear (bool): Whether to auto-clear when threshold is exceeded
        """
        self.threshold_percent = threshold_percent
        self.auto_clear = auto_clear
        self.peak_memory = 0
        self.start_memory = 0
        self.tracking = False
    
    def start_tracking(self) -> None:
        """Start tracking GPU memory usage"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available - cannot track GPU memory")
            return
        
        try:
            self.tracking = True
            self.start_memory = torch.cuda.memory_allocated(0)
            self.peak_memory = self.start_memory
            logger.info(f"Started GPU memory tracking. Initial memory: {self.start_memory / 1024**2:.2f} MB")
        except Exception as e:
            logger.error(f"Error starting GPU memory tracking: {e}")
    
    def check_memory(self) -> Tuple[float, bool]:
        """
        Check current memory usage and clear if needed.
        
        Returns:
            Tuple[float, bool]: Current memory percentage and whether memory was cleared
        """
        if not torch.cuda.is_available() or not self.tracking:
            return 0.0, False
        
        try:
            memory_status = get_gpu_memory_status()
            
            if memory_status is None:
                return 0.0, False
            
            current_memory = memory_status.used_memory
            memory_percent = memory_status.utilization_percent
            
            # Update peak memory
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
            
            # Clear memory if threshold is exceeded
            if memory_percent > self.threshold_percent and self.auto_clear:
                logger.warning(f"GPU memory utilization ({memory_percent:.2f}%) exceeded threshold "
                               f"({self.threshold_percent:.2f}%). Clearing memory...")
                cleared = clear_gpu_memory()
                return memory_percent, cleared
            
            return memory_percent, False
        except Exception as e:
            logger.error(f"Error checking GPU memory: {e}")
            return 0.0, False
    
    def stop_tracking(self) -> Dict[str, float]:
        """
        Stop tracking and return statistics.
        
        Returns:
            Dict[str, float]: Memory usage statistics
        """
        if not torch.cuda.is_available() or not self.tracking:
            return {"peak_mb": 0, "final_mb": 0, "diff_mb": 0}
        
        try:
            final_memory = torch.cuda.memory_allocated(0)
            
            stats = {
                "peak_mb": self.peak_memory / 1024**2,
                "final_mb": final_memory / 1024**2,
                "diff_mb": (final_memory - self.start_memory) / 1024**2
            }
            
            logger.info(f"Stopped GPU memory tracking. Peak memory: {stats['peak_mb']:.2f} MB")
            
            self.tracking = False
            return stats
        except Exception as e:
            logger.error(f"Error stopping GPU memory tracking: {e}")
            return {"peak_mb": 0, "final_mb": 0, "diff_mb": 0}


def optimize_batch_size(
    sample_size: int,
    model_fn: callable,
    initial_batch_size: int = 32,
    max_memory_percent: float = 80.0,
    min_batch_size: int = 4
) -> int:
    """
    Find optimal batch size for a given model function.
    
    Args:
        sample_size (int): Size of input to use for testing
        model_fn (callable): Function that takes a batch size parameter
        initial_batch_size (int): Initial batch size to try
        max_memory_percent (float): Maximum memory utilization percent
        min_batch_size (int): Minimum acceptable batch size
        
    Returns:
        int: Optimal batch size
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available - cannot optimize batch size")
        return initial_batch_size
    
    # Clear memory before starting
    clear_gpu_memory()
    
    batch_size = initial_batch_size
    tracker = GpuMemoryTracker(threshold_percent=max_memory_percent, auto_clear=False)
    
    try:
        # Try with initial batch size
        tracker.start_tracking()
        model_fn(batch_size)
        memory_stats = tracker.stop_tracking()
        
        # If memory usage is too high, decrease batch size
        memory_percent = memory_stats.get("peak_mb", 0) / get_gpu_memory_status().total_memory * 1024**2 * 100
        
        if memory_percent > max_memory_percent:
            # Decrease batch size until it fits
            while batch_size > min_batch_size and memory_percent > max_memory_percent:
                batch_size = max(min_batch_size, batch_size // 2)
                
                # Clear memory
                clear_gpu_memory()
                
                # Try with new batch size
                tracker.start_tracking()
                model_fn(batch_size)
                memory_stats = tracker.stop_tracking()
                
                memory_percent = memory_stats.get("peak_mb", 0) / get_gpu_memory_status().total_memory * 1024**2 * 100
        
        logger.info(f"Optimized batch size: {batch_size}")
        return batch_size
    except Exception as e:
        logger.error(f"Error optimizing batch size: {e}")
        # Fall back to a reasonable batch size
        return min(initial_batch_size, 16)
