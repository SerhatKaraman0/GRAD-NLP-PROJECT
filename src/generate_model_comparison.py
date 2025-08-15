#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script generates a comparison dashboard for all available NLP models.
It compares performance metrics and visualizes the differences between models.
"""

import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Now import the modules
from src.core.common_imports import * # noqa: F403, F405
from src.core.nlpmodel import NlpModel
from src.visualization.dashboard_generator import DashboardGenerator

import os
import logging
import argparse
import webbrowser

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate model comparison dashboard')
    return parser.parse_args()

def main():
    """Main function to generate model comparison dashboard"""
    args = parse_args()
    
    # Initialize NlpModel to get base directories
    nlp_model = NlpModel()
    base_dir = nlp_model.BASE_DIR
    save_data_dir = os.path.join(base_dir, "data")
    metrics_dir = os.path.join(save_data_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Initialize logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Generating model comparison dashboard")
    
    # Initialize dashboard generator
    logger.info("Initializing Dashboard Generator...")
    dashboard_generator = DashboardGenerator(save_data_dir=save_data_dir)
    
    # Generate model comparison dashboard
    logger.info("Generating model comparison dashboard...")
    dashboard_path = dashboard_generator.generate_model_comparison_dashboard()
    
    if dashboard_path:
        logger.info(f"Model comparison dashboard generated at: {dashboard_path}")
        
        # Open the dashboard
        try:
            webbrowser.open('file://' + dashboard_path)
            logger.info("Dashboard opened in web browser")
        except:
            logger.warning("Could not open dashboard in web browser")
    else:
        logger.error("Failed to generate model comparison dashboard")

if __name__ == "__main__":
    main()
