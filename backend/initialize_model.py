#!/usr/bin/env python3
"""
Model Initialization Script for Edu- Project

This script ensures the ML recommendation model is properly trained
and available before deployment. Run this script before deploying
the application.

Usage:
    python backend/initialize_model.py
"""

import os
import sys
from pathlib import Path
import json

# Add the backend directory to the path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from model_trainer import train_model

def check_model_exists():
    """Check if the model file exists and is valid"""
    model_path = current_dir / "models" / "recommender_model.pkl"
    
    if not os.path.exists(model_path):
        return False, "Model file does not exist"
    
    try:
        import pickle
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        
        # Check if all required components are present
        required_keys = ["model", "vectorizer", "data"]
        for key in required_keys:
            if key not in saved:
                return False, f"Model file missing {key} component"
        
        return True, "Model file is valid"
    except Exception as e:
        return False, f"Model file is corrupted: {e}"

def initialize_model():
    """Initialize the ML model for the application"""
    print("🚀 Initializing Edu- ML Model...")
    print("=" * 50)
    
    # Check if model already exists
    model_exists, message = check_model_exists()
    
    if model_exists:
        print(f"✅ {message}")
        print("📊 Model is ready for deployment!")
        return True
    
    print(f"⚠️  {message}")
    print("🔄 Training new model...")
    
    # Train the model
    try:
        result = train_model()
        
        if result["status"] == "success":
            print("✅ Model training completed successfully!")
            print(f"📈 Dataset size: {result['dataset_size']} resources")
            print(f"🔢 Features: {result['features']}")
            print(f"📚 Domains: {result['domains']}")
            print(f"📖 Subjects: {result['subjects']}")
            print(f"💾 Model saved at: {result['model_path']}")
            return True
        else:
            print(f"❌ Model training failed: {result['message']}")
            return False
            
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        return False

def verify_deployment_readiness():
    """Verify that the application is ready for deployment"""
    print("\n🔍 Verifying deployment readiness...")
    print("=" * 50)
    
    checks = []
    
    # Check 1: Model file exists
    model_exists, _ = check_model_exists()
    checks.append(("ML Model", model_exists))
    
    # Check 2: Dataset exists
    data_path = current_dir.parent / "data" / "resources_curated.csv"
    dataset_exists = os.path.exists(data_path)
    checks.append(("Dataset", dataset_exists))
    
    # Check 3: Required directories exist
    models_dir = current_dir / "models"
    models_dir_exists = os.path.exists(models_dir)
    checks.append(("Models Directory", models_dir_exists))
    
    # Display results
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 Application is ready for deployment!")
        return True
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    print("Edu- Model Initialization")
    print("=" * 50)
    
    # Initialize the model
    success = initialize_model()
    
    if success:
        # Verify deployment readiness
        verify_deployment_readiness()
    else:
        print("\n❌ Model initialization failed. Please check the errors above.")
        sys.exit(1)
