#!/usr/bin/env python
# filepath: /Users/user/Desktop/Projects/NLP-Learning/src/features/download_embeddings.py

"""
Script to download and extract GloVe embeddings if they don't already exist.
This ensures the pre-trained embeddings are available before running the model training.
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Import project modules
from src.core.logging_config import setup_logging, get_logger

# Setup logger
setup_logging()
logger = get_logger('Download_Embeddings')

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """
    Download a file from URL with progress bar
    """
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def ensure_glove_embeddings():
    """
    Download and extract GloVe embeddings if they don't exist
    """
    # Define paths
    embeddings_dir = os.path.join(project_root, 'data', 'embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)
    
    glove_zip_path = os.path.join(embeddings_dir, 'glove.6B.zip')
    
    # Check if embeddings already exist
    dimensions = [50, 100, 200, 300]
    embedding_files = [os.path.join(embeddings_dir, f'glove.6B.{d}d.txt') for d in dimensions]
    
    if all(os.path.exists(f) for f in embedding_files):
        logger.info("All GloVe embeddings already available")
        return True
    
    # Download embeddings if not already downloaded
    if not os.path.exists(glove_zip_path):
        logger.info("Downloading GloVe embeddings (862MB)...")
        try:
            download_url('http://nlp.stanford.edu/data/glove.6B.zip', glove_zip_path)
            logger.info("Download complete")
        except Exception as e:
            logger.error(f"Error downloading GloVe embeddings: {e}")
            return False
    else:
        logger.info("GloVe embeddings zip already downloaded")
    
    # Extract embeddings
    logger.info("Extracting GloVe embeddings...")
    try:
        with zipfile.ZipFile(glove_zip_path, 'r') as zip_ref:
            zip_ref.extractall(embeddings_dir)
        logger.info("Extraction complete")
        return True
    except Exception as e:
        logger.error(f"Error extracting GloVe embeddings: {e}")
        return False

def main():
    """
    Main function to ensure embeddings are downloaded
    """
    logger.info("Checking GloVe embeddings availability...")
    success = ensure_glove_embeddings()
    
    if success:
        logger.info("GloVe embeddings ready to use")
    else:
        logger.error("Failed to setup GloVe embeddings")
        sys.exit(1)

if __name__ == "__main__":
    main()
