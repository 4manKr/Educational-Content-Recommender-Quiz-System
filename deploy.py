#!/usr/bin/env python3
"""
Deployment Script for Edu- Project

This script prepares the application for deployment by:
1. Installing dependencies
2. Initializing the ML model
3. Verifying all components are ready
4. Starting the application

Usage:
    python deploy.py
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"[RUNNING] {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"[SUCCESS] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"[ERROR] Python {version.major}.{version.minor} is not supported. Please use Python 3.8 or higher.")
        return False
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    if not os.path.exists("requirements.txt"):
        print("[ERROR] requirements.txt not found")
        return False
    
    return run_command("pip install -r requirements.txt", "Installing dependencies")

def initialize_model():
    """Initialize the ML model"""
    return run_command("python backend/initialize_model.py", "Initializing ML model")

def start_application():
    """Start the Streamlit application"""
    print("Starting Edu- application...")
    print("The application will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        subprocess.run("streamlit run frontend/app.py", shell=True, check=True)
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to start application: {e}")
        return False
    
    return True

def main():
    """Main deployment function"""
    print("Edu- Deployment Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("[ERROR] Failed to install dependencies")
        sys.exit(1)
    
    # Initialize model
    if not initialize_model():
        print("[ERROR] Failed to initialize model")
        sys.exit(1)
    
    # Start application
    start_application()

if __name__ == "__main__":
    main()
