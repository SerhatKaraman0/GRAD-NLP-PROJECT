"""
Launcher script for the NLP model testing interface.
This script checks requirements and starts the Flask server for the web interface.
Ensures that the server only starts after all models are properly loaded.
"""

import os
import sys
import subprocess
import webbrowser
import json
from time import sleep

def check_requirements():
    """Check if required packages are installed"""
    try:
        import flask
        import pandas
        import torch
        import requests
        return True
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages with: pip install flask pandas torch requests")
        return False

def main():
    """Run the test interface"""
    print("=== NLP Model Testing Interface ===")
    
    # Check requirements
    print("Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to project root directory
    os.chdir(os.path.join(script_dir, ".."))
    
    # Start the Flask server
    print("\nStarting test server...")
    print("Loading models... This may take a moment...")
    server_process = subprocess.Popen(
        [sys.executable, os.path.join(script_dir, "model_test_server.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start and check health endpoint
    max_attempts = 30  # Increased for more patience with model loading
    attempts = 0
    server_ready = False
    components_loaded = False
    models_loaded = False
    print("Waiting for server to initialize...")
    
    while attempts < max_attempts and not server_ready:
        try:
            import requests
            response = requests.get('http://localhost:8000/health')
            if response.status_code == 200:
                data = response.json()
                components_loaded = data.get('components_loaded', False)
                models_loaded = data.get('models_loaded', False)
                
                if components_loaded and models_loaded:
                    server_ready = True
                    print("\n✅ Server is ready! All models loaded successfully.")
                elif components_loaded:
                    # Components loaded but still waiting on models
                    model_status = data.get('models', {})
                    loaded = [m for m, status in model_status.items() if status]
                    not_loaded = [m for m, status in model_status.items() if not status]
                    
                    if not attempts % 3:  # Only print this every 3 attempts to avoid spam
                        print("\nLoading models:")
                        for model, status in model_status.items():
                            status_text = "✅ Loaded" if status else "⏳ Loading..."
                            print(f"  - {model.capitalize()}: {status_text}")
                else:
                    if not attempts % 3:
                        print("\nInitializing components...")
            else:
                print(".", end="", flush=True)
        except Exception:
            print(".", end="", flush=True)
        
        sleep(1.5)
        attempts += 1
    
    if not server_ready:
        print("\n❌ Server failed to start properly or timed out loading models.")
        print("Check logs for details or try again.")
        server_process.terminate()
        sys.exit(1)
        
    # Open the browser
    print("\n🌐 Opening test interface in browser...")
    webbrowser.open("http://localhost:8000")
    
    print("\n⚠️  Server is running. Press Ctrl+C to stop.")
    try:
        # Keep the script running
        server_process.wait()
    except KeyboardInterrupt:
        # Stop the server when user presses Ctrl+C
        print("\n🛑 Stopping server...")
        server_process.terminate()
        server_process.wait()
        print("✅ Server stopped.")

if __name__ == "__main__":
    main()
