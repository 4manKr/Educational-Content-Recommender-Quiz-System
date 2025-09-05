import pandas as pd
from pathlib import Path


def recommend_from_weakness(weak_domains, weak_topics, n=3):
    """
    Recommend resources for weak domains and topics.
    """
    # Get the correct path relative to the current file location
    current_dir = Path(__file__).parent
    data_path = current_dir.parent / "data" / "resources_cleaned.csv"
    resources = pd.read_csv(data_path)

    recommendations = {}

    # Domain-based recommendations
    for domain in weak_domains:
        subset = resources[resources['domain'] == domain]
        if not subset.empty:
            recs = subset.sample(min(n, len(subset)))
            recommendations[f"Domain: {domain}"] = recs[['title', 'url', 'type', 'level_availability']].to_dict(orient="records")

    # Topic-based recommendations (search in title/subjects)
    for topic in weak_topics.keys():
        subset = resources[resources['title'].str.contains(topic, case=False, na=False) |
                           resources['subjects'].str.contains(topic, case=False, na=False)]
        if not subset.empty:
            recs = subset.sample(min(n, len(subset)))
            recommendations[f"Topic: {topic}"] = recs[['title', 'url', 'type', 'level_availability']].to_dict(orient="records")

    return recommendations
