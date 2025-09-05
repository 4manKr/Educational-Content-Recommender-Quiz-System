import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle
import os
import json
from pathlib import Path

def train_model(data_path=None, model_path=None, max_features=5000, n_neighbors=5):
    """
    Train the recommendation model and return training metrics
    
    Args:
        data_path: Path to the dataset CSV file
        model_path: Path to save the trained model
        max_features: Maximum features for TF-IDF vectorizer
        n_neighbors: Number of neighbors for NearestNeighbors
    
    Returns:
        dict: Training metrics and status
    """
    try:
        # Set default paths if not provided
        if data_path is None:
            current_dir = Path(__file__).parent
            data_path = current_dir.parent / "data" / "resources_curated.csv"
        
        if model_path is None:
            current_dir = Path(__file__).parent
            model_path = current_dir / "models" / "recommender_model.pkl"
        
        # Step 1: Load dataset
        print("📂 Loading dataset...")
        df = pd.read_csv(data_path)
        
        # Check basic info
        print("✅ Dataset loaded:", df.shape)
        
        # Step 2: Preprocess text
        print("🛠 Preprocessing...")
        df["combined"] = df["domain"].astype(str) + " " + df["subjects"].astype(str) + " " + df["title"].astype(str)
        
        # Step 3: Vectorize
        print("🔢 Vectorizing resources...")
        vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
        X = vectorizer.fit_transform(df["combined"])
        
        print("✅ TF-IDF shape:", X.shape)
        
        # Step 4: Train Nearest Neighbors model
        print("🤖 Training recommender model...")
        model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        model.fit(X)
        
        print("✅ Model trained successfully!")
        
        # Step 5: Save model + vectorizer
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "vectorizer": vectorizer, "data": df}, f)
        
        print(f"💾 Model saved at {model_path}")
        
        # Return training metrics
        return {
            "status": "success",
            "dataset_size": df.shape[0],
            "features": X.shape[1],
            "model_path": str(model_path),
            "domains": df["domain"].nunique(),
            "subjects": df["subjects"].nunique(),
            "message": "Model trained successfully!"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Training failed: {str(e)}"
        }

if __name__ == "__main__":
    # Run training when script is executed directly
    result = train_model()
    print(json.dumps(result, indent=2))
