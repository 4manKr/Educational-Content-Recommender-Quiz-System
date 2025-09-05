import pickle
import os
from pathlib import Path

# Global variables for model components
model = None
vectorizer = None
data = None

def load_model():
    """Load the trained model and components"""
    global model, vectorizer, data
    
    try:
        # Get the correct path relative to this file
        current_dir = Path(__file__).parent
        model_path = current_dir / "models" / "recommender_model.pkl"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        
        model = saved["model"]
        vectorizer = saved["vectorizer"]
        data = saved["data"]
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def recommend_resources(query, top_k=5):
    """Given a query (weak subject), return recommended resources."""
    global model, vectorizer, data
    
    # Load model if not already loaded
    if model is None:
        if not load_model():
            return []
    
    try:
        vec = vectorizer.transform([query])
        distances, indices = model.kneighbors(vec, n_neighbors=top_k)
        results = data.iloc[indices[0]][["domain", "subjects", "title", "type", "url"]]
        return results.to_dict(orient="records")
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return []

# Example usage
if __name__ == "__main__":
    recs = recommend_resources("algebra equations")
    for r in recs:
        print(f"- {r['title']} ({r['url']})")
