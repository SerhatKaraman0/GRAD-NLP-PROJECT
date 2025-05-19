"""
Flask server for testing NLP models through a web interface.
This server handles both single text predictions and batch CSV predictions.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory
import csv
from io import StringIO

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import required modules from the project
from src.core.logging_config import * # noqa: F403, F405
from src.models.model_builder import ModelBuilder
from src.features.embeddings import EmbeddingProcessor

app = Flask(__name__)

# Initialize the logger
logger = logging.getLogger('ModelTestServer')

# Initialize key components
save_data_dir = os.path.join(project_root, "data")
models_dir = os.path.join(save_data_dir, "models")
embedding_processor = None
model_builder = None

# Model loading status tracking
model_loading_status = {
    'simple': False,
    'deep': False,
    'stacked': False,
    'ensemble': False
}

def load_components():
    """Initialize necessary components for model testing"""
    global embedding_processor, model_builder
    
    try:
        # Initialize with reasonable defaults
        embedding_processor = EmbeddingProcessor(
            batch_size=1000,
            max_features=10000,
            embedding_dim=100
        )
        model_builder = ModelBuilder(models_dir)
        logger.info("Components loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading components: {e}")
        return False

def preprocess_text(text):
    """Preprocess text for model input"""
    try:
        # Validate input
        if not text or not isinstance(text, str):
            logger.error(f"Invalid text input: {text}")
            raise ValueError(f"Invalid text input type: {type(text)}")
            
        # Convert to sequence and pad
        logger.debug(f"Preprocessing text: {text[:50]}...")
        sequences, _ = embedding_processor.prepare_embeddings([text])
        
        if not sequences or len(sequences) == 0 or len(sequences[0]) == 0:
            logger.warning(f"Empty sequence after processing: {text[:50]}...")
            # Return a default tensor with padding token (0) if sequence is empty
            return torch.tensor([[0]], dtype=torch.long)
        
        max_len = min(len(sequences[0]), 500)  # Cap at 500 tokens
        X = torch.tensor(sequences[0][:max_len], dtype=torch.long).unsqueeze(0)
        return X
    except Exception as e:
        import traceback
        logger.error(f"Error preprocessing text: {e}\n{traceback.format_exc()}")
        raise

def predict_single(text, model_type='simple', load_best=True):
    """Predict sentiment for a single text input"""
    try:
        # Preprocess the text
        X = preprocess_text(text)
        
        # Get embedding matrix size
        vocab_size = embedding_processor.max_features + 1
        
        # Load the model
        model = model_builder.load_model(
            input_size=vocab_size,
            embedding_dim=embedding_processor.embedding_dim,
            model_type=model_type,
            load_best=load_best
        )
        
        # Move model to CPU and set to evaluation mode
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        # Make prediction
        with torch.no_grad():
            output = model(X)
            predicted_class = torch.argmax(output, dim=1).item()  # 0-4
            prediction = predicted_class + 1  # Convert to 1-5 stars
            
        return float(prediction)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

@app.route('/')
def index():
    """Serve the test interface HTML page"""
    return send_from_directory(os.path.dirname(__file__), 'model_test_interface.html')

@app.route('/predict_text', methods=['POST'])
def predict_text_endpoint():
    """Endpoint for single text prediction"""
    try:
        # Get request data
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        model_type = data.get('model_type', 'simple')
        load_best = data.get('load_best', True)
        
        logger.info(f"Processing text prediction: model={model_type}, load_best={load_best}")
        
        # Make prediction
        prediction = predict_single(text, model_type, load_best)
        
        logger.info(f"Prediction successful: {prediction}")
        
        return jsonify({
            'text': text,
            'prediction': prediction,
            'model_type': model_type
        })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error in predict_text endpoint: {e}\n{error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 500

@app.route('/predict_csv', methods=['POST'])
def predict_csv_endpoint():
    """Endpoint for batch CSV prediction"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Get other parameters
        model_type = request.form.get('model_type', 'simple')
        load_best = request.form.get('load_best', 'true') == 'true'
        
        logger.info(f"Processing CSV prediction: model={model_type}, load_best={load_best}")
        
        # Read the CSV file
        csv_data = file.read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_data))
        
        logger.info(f"CSV loaded successfully with {len(df)} rows")
        
        # Check if 'Text' column exists
        if 'Text' not in df.columns:
            logger.warning(f"CSV columns: {df.columns.tolist()}. 'Text' column not found.")
            return jsonify({'error': 'CSV must contain a "Text" column'}), 400
        
        # Process each text (limit to first 50 for performance)
        results = []
        errors = []
        
        for idx, row in df.head(50).iterrows():
            if pd.isna(row['Text']) or row['Text'].strip() == '':
                continue
            
            try:
                prediction = predict_single(row['Text'], model_type, load_best)
                results.append({
                    'text': row['Text'][:100] + ('...' if len(row['Text']) > 100 else ''),  # Truncate for display
                    'prediction': prediction
                })
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                errors.append({'row': idx, 'error': str(e)})
        
        logger.info(f"CSV processing complete: {len(results)} successful, {len(errors)} failed")
        
        return jsonify({
            'predictions': results,
            'model_type': model_type,
            'total_processed': len(results),
            'errors': errors
        })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error in predict_csv endpoint: {e}\n{error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 500

def load_all_models():
    """Load all models to ensure they're available before starting"""
    try:
        logger.info("Pre-loading all models...")
        model_types = ['simple', 'deep', 'stacked', 'ensemble']
        load_options = [True, False]  # best and latest
        
        # Get embedding matrix size for models
        vocab_size = embedding_processor.max_features + 1
        embedding_dim = embedding_processor.embedding_dim
        
        all_loaded_successfully = True
        
        for model_type in model_types:
            model_loading_status[model_type] = False
            try:
                # Try to load both best and latest versions
                for load_best in load_options:
                    logger.info(f"Loading model: {model_type} (best={load_best})")
                    model = model_builder.load_model(
                        input_size=vocab_size,
                        embedding_dim=embedding_dim,
                        model_type=model_type,
                        load_best=load_best
                    )
                    if model is None:
                        logger.warning(f"Model {model_type} (best={load_best}) could not be loaded")
                        all_loaded_successfully = False
                        break
                
                # If we got here without breaking, the model was loaded successfully
                model_loading_status[model_type] = True
                logger.info(f"Model {model_type} loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_type}: {e}")
                model_loading_status[model_type] = False
                all_loaded_successfully = False
        
        if all_loaded_successfully:
            logger.info("All models loaded successfully")
        else:
            logger.warning("Some models could not be loaded")
            
        return all_loaded_successfully
    except Exception as e:
        logger.error(f"Error during model preloading: {e}")
        return False

def init_server():
    """Initialize server components and models"""
    if not load_components():
        logger.error("Failed to load required components. Exiting.")
        return False
        
    if not load_all_models():
        logger.error("Failed to load all required models. Exiting.")
        return False
        
    logger.info("Server initialization complete")
    return True

# Server healthcheck endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint for server health check"""
    all_models_loaded = all(model_loading_status.values())
    return jsonify({
        'status': 'ok',
        'components_loaded': embedding_processor is not None and model_builder is not None,
        'models_loaded': all_models_loaded,
        'models': model_loading_status
    })

if __name__ == '__main__':
    # Initialize server before running
    if init_server():
        # Run the Flask app
        port = int(os.environ.get("PORT", 8000))
        logger.info(f"Starting server on port {port}")
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        logger.critical("Server initialization failed. Cannot start server.")
        sys.exit(1)
