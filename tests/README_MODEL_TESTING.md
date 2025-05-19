# Model Testing Interface

This directory contains files for manually testing the NLP sentiment analysis models through a web interface.

## Features

- Test any of the available models (Simple LSTM, Deep LSTM, Stacked LSTM, Ensemble)
- Input individual sentences for quick predictions
- Upload CSV files for batch processing
- View and compare model predictions

## Files

- `model_test_interface.html`: The HTML interface for testing models
- `model_test_server.py`: Flask server that handles prediction requests
- `run_model_test_interface.py`: Launcher script to start the testing interface

## How to Use

1. Make sure you have the required dependencies installed:
   ```
   pip install flask pandas torch
   ```

2. Run the interface with:
   ```
   python run_model_test_interface.py
   ```

3. Your web browser will open automatically with the testing interface.

4. Select a model type using the radio buttons.

5. For single sentence testing:
   - Enter your text in the provided text area
   - Click "Predict Rating"

6. For batch CSV testing:
   - Prepare a CSV file with a column named "Text" containing reviews
   - Upload the CSV file using the file input
   - Click "Process CSV"

7. View the prediction results displayed on the page.

## Notes

- The interface is for testing purposes only.
- Large CSV files may take time to process.
- For best results, make sure trained models exist in the data/models directory.
