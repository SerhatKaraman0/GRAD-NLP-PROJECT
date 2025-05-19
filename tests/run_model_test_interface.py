"""
Launcher script for the NLP model testing interface.
This script checks requirements and starts the Flask server for the web interface.
"""

import os
import sys
import subprocess
import webbrowser
from time import sleep

def check_requirements():
    """Check if required packages are installed"""
    try:
        import flask
        import pandas
        import torch
        return True
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages with: pip install flask pandas torch")
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
    server_process = subprocess.Popen(
        [sys.executable, os.path.join(script_dir, "model_test_server.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait briefly for server to start
    print("Waiting for server to start...")
    sleep(2)
    
    # Open the browser
    print("Opening test interface in browser...")
    webbrowser.open("http://localhost:8000")
    
    print("\nServer is running. Press Ctrl+C to stop.")
    try:
        # Keep the script running
        server_process.wait()
    except KeyboardInterrupt:
        # Stop the server when user presses Ctrl+C
        print("\nStopping server...")
        server_process.terminate()
        server_process.wait()
        print("Server stopped.")

if __name__ == "__main__":
    main()
