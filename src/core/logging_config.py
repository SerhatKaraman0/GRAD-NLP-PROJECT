import logging
import os

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logging(log_level=logging.INFO):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(LOG_DIR, "app.log")),
            logging.StreamHandler()
        ]
    )

def get_logger(name):
    """
    Get a logger with the specified name
    
    Args:
        name: Name for the logger
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)