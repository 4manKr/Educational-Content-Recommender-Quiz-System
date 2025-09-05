import pickle

MODEL_PATH = "../backend/models/recommender_model.pkl"

# Load model once
with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
vectorizer = saved["vectorizer"]
data = saved["data"]

def recommend_resources(query, top_k=5):
    """Given a query (weak subject), return recommended resources."""
    vec = vectorizer.transform([query])
    distances, indices = model.kneighbors(vec, n_neighbors=top_k)
    results = data.iloc[indices[0]][["domain", "subjects", "title", "type", "url"]]
    return results.to_dict(orient="records")

# Example usage
if __name__ == "__main__":
    recs = recommend_resources("algebra equations")
    for r in recs:
        print(f"- {r['title']} ({r['url']})")
