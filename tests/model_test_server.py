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
        device = torch.device('cpu')
        model.to(device)
        model.eval()
        
        # Make prediction
        with torch.no_grad():
            output = model(X)
            # Get the scalar value and then clamp between 1-5
            output_value = output.item()
            prediction = max(1.0, min(5.0, output_value))
            
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

@app.before_first_request
def initialize_components():
    """Ensure components are loaded before handling the first request"""
    global embedding_processor, model_builder
    if embedding_processor is None or model_builder is None:
        load_components()

if __name__ == '__main__':
    # Load necessary components
    if load_components():
        # Run the Flask app
        port = int(os.environ.get("PORT", 8000))
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        print("Failed to initialize components, server cannot start.")
