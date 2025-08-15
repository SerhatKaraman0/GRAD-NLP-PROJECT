# Model Testing Interface

This directory contains files for manually testing the NLP sentiment analysis models through a modern, intuitive web interface.

## Features

- **Modern UI**: Clean and responsive design with visual feedback
- Test any of the available models (Simple LSTM, Deep LSTM, Stacked LSTM, Ensemble)
- Input individual sentences for quick predictions
- Upload CSV files for batch processing
- View and compare model predictions
- **Loading Indicators**: Visual feedback during model loading and predictions
- **Proper Model Loading**: Server ensures all models are loaded before accepting requests

## Files

- `model_test_interface.html`: The HTML interface for testing models
- `model_test_server.py`: Flask server that handles prediction requests
- `run_model_test_interface.py`: Launcher script to start the testing interface

## How to Use

1. Make sure you have the required dependencies installed:
   ```
   pip install flask pandas torch requests
   ```

2. Run the interface with:
   ```
   python tests/run_model_test_interface.py
   ```

3. **Wait for Models to Load**:
   - The interface shows a loading screen while models are initialized
   - Individual model loading status is displayed
   - Interface becomes available only when all models are ready

4. Your web browser will open automatically when the models are loaded and ready.

5. Select a model type using the radio buttons.

6. For single sentence testing:
   - Enter your text in the provided text area
   - Click "Predict Rating"

7. For batch CSV testing:
   - Prepare a CSV file with a column named "Text" containing reviews
   - Upload the CSV file using the file input
   - Click "Process CSV"

8. View the prediction results displayed on the page with star ratings visualization.

## Troubleshooting

If you encounter any issues:

1. **Models Fail to Load**: 
   - Check that model files exist in the `/data/models/` directory
   - Ensure model files are compatible with the current code version

2. **Server Startup Issues**:
   - Verify all required Python packages are installed
   - Check the server logs for specific error messages

3. **Connection Errors**:
   - Ensure port 8000 is available and not used by other applications

## Notes

- The interface is for testing purposes only.
- CSV processing is limited to 50 rows at a time for performance.
- For best results, make sure trained models exist in the data/models directory.
- To stop the server, press Ctrl+C in the terminal where you launched the script.
