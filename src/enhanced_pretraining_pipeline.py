#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script enhances the pretraining pipeline to include rating-based visualizations.
It processes the review data, generates visualizations based on ratings,
and prepares the data for model training.
"""

import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Now import the modules
from src.core.common_imports import * # noqa: F403, F405
from src.core.nlpmodel import NlpModel
from src.data.data_processing import DataProcessor
from src.visualization.dashboard_generator import DashboardGenerator

import os
import argparse
import logging
import time

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced pretraining pipeline with rating visualizations')
    parser.add_argument('--max_features', type=int, default=10000,
                        help='Maximum number of features for vectorization')
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='Batch size for processing')
    parser.add_argument('--skip_visuals', action='store_true',
                        help='Skip generating visualizations')
    parser.add_argument('--generate_comparison', action='store_true',
                        help='Generate model comparison dashboard')
    return parser.parse_args()

def main():
    """Main function to run the enhanced pretraining pipeline"""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Start timer
    start_time = time.time()
    
    # Initialize NlpModel to get base directories
    nlp_model = NlpModel()
    base_dir = nlp_model.BASE_DIR
    save_data_dir = os.path.join(base_dir, "data")
    
    logger.info("=== Enhanced Pretraining Pipeline ===")
    logger.info(f"Max features: {args.max_features}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Create data processor
    logger.info("Initializing Data Processor...")
    data_processor = DataProcessor(
        batch_size=args.batch_size,
        max_features=args.max_features
    )
    
    # Load and process data
    logger.info("Loading and processing data...")
    texts, labels = data_processor.load_and_process_data()
    logger.info(f"Processed {len(texts)} reviews")
    
    # Generate bag of words representation
    logger.info("Creating bag of words representation...")
    bow_df = data_processor.create_bow()
    logger.info(f"Created BoW with shape {bow_df.shape}")
    
    # Calculate basic word frequency
    logger.info("Calculating word frequency...")
    data_processor.word_freq()
    
    # Generate rating visualizations if not skipped
    if not args.skip_visuals:
        logger.info("Generating rating visualizations...")
        visualizations = data_processor.generate_rating_visualizations()
        logger.info(f"Generated visualizations: {list(visualizations.keys())}")
        
        # Initialize dashboard generator with sample metrics
        # In a real scenario, these would come from trained models
        logger.info("Creating sample dashboard...")
        dashboard_generator = DashboardGenerator(save_data_dir=save_data_dir)
        
        # Sample model results for demonstration
        sample_results = {
            'mse': 0.5,
            'mae': 0.6,
            'accuracy': 0.85,
            'precision': 0.84,
            'recall': 0.82,
            'f1': 0.83,
            'r2': 0.75
        }
        
        dashboard_path = dashboard_generator.generate_model_dashboard(
            model_results=sample_results,
            model_type='ensemble'
        )
        logger.info(f"Sample dashboard generated at: {dashboard_path}")
    
    # Generate model comparison dashboard if requested
    if args.generate_comparison:
        logger.info("Generating model comparison dashboard...")
        # For demonstration, using the same sample_results
        # In practice, this would compare different models' results
        comparison_dashboard_path = dashboard_generator.generate_model_comparison_dashboard(
            model_results=[sample_results, sample_results],  # Dummy comparison
            model_names=['Model A', 'Model B']
        )
        logger.info(f"Model comparison dashboard generated at: {comparison_dashboard_path}")
    
    # End timer
    end_time = time.time()
    logger.info(f"Pipeline completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
