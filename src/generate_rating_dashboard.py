#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script generates visualizations for reviews by ratings and creates a dashboard.
It demonstrates the enhanced functionality added to the project.
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
import logging
import argparse

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate rating visualizations dashboard')
    parser.add_argument('--model_type', type=str, default='ensemble',
                        choices=['simple', 'deep', 'stacked', 'ensemble'],
                        help='Model type to generate dashboard for')
    return parser.parse_args()

def main():
    """Main function to generate rating visualizations and dashboard"""
    args = parse_args()
    model_type = args.model_type
    
    # Initialize NlpModel to get base directories
    nlp_model = NlpModel()
    base_dir = nlp_model.BASE_DIR
    save_data_dir = os.path.join(base_dir, "data")
    metrics_dir = os.path.join(save_data_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Initialize logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"Generating rating visualizations and dashboard for {model_type} model")
    
    # Create Data Processor
    logger.info("Initializing Data Processor...")
    data_processor = DataProcessor()
    
    # Generate rating visualizations
    logger.info("Generating rating visualizations...")
    visualizations = data_processor.generate_rating_visualizations()
    logger.info(f"Generated visualizations: {list(visualizations.keys())}")
    
    # Create sample model results for dashboard (if not loading actual results)
    model_results = {
        'mse': 0.5,
        'mae': 0.6,
        'accuracy': 0.85,
        'precision': 0.84,
        'recall': 0.82,
        'f1': 0.83,
        'r2': 0.75
    }
    
    # Initialize dashboard generator
    logger.info("Initializing Dashboard Generator...")
    dashboard_generator = DashboardGenerator(save_data_dir=save_data_dir)
    
    # Generate dashboard
    logger.info("Generating dashboard...")
    dashboard_path = dashboard_generator.generate_model_dashboard(
        model_results=model_results,
        model_type=model_type
    )
    
    logger.info(f"Dashboard generated at: {dashboard_path}")
    
    # Open the dashboard
    try:
        import webbrowser
        webbrowser.open('file://' + dashboard_path)
        logger.info("Dashboard opened in web browser")
    except:
        logger.warning("Could not open dashboard in web browser")

if __name__ == "__main__":
    main()
