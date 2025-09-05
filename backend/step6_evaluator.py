import json
import re
import pandas as pd
from pathlib import Path


def evaluate_quiz(quiz_json_path, user_answers):
    """
    Evaluates quiz answers against the dataset.
    Returns per-question results, domain scores, weak domains, and weak topics.
    """

    # Load dataset
    # Get the correct path relative to the current file location
    current_dir = Path(__file__).parent
    data_path = current_dir.parent / "data" / "resources_cleaned.csv"
    df = pd.read_csv(data_path)

    # Load quiz
    with open(quiz_json_path, "r") as f:
        quiz = json.load(f)

    per_q = []
    domain_scores = {}
    weak_topics = {}

    for i, q in enumerate(quiz):
        q_text = q["question"]
        correct = q["answer"].strip().lower()
        user_ans = user_answers[i].strip().lower()

        domain = q.get("domain", "General")
        topic = q.get("topic", "Unknown")

        # Check correctness
        is_correct = (correct == user_ans)

        # Track domain score
        if domain not in domain_scores:
            domain_scores[domain] = [0, 0]  # [correct, total]
        domain_scores[domain][1] += 1
        if is_correct:
            domain_scores[domain][0] += 1

        # Track weak topics if wrong
        if not is_correct:
            if topic not in weak_topics:
                weak_topics[topic] = 0
            weak_topics[topic] += 1

        # Save per-question result
        per_q.append({
            "question": q_text,
            "correct": correct,
            "user": user_ans,
            "is_correct": is_correct,
            "domain": domain,
            "topic": topic
        })

    # Compute weak domains (score < 50%)
    weak_domains = []
    for d, (c, t) in domain_scores.items():
        if t > 0 and (c / t) < 0.5:
            weak_domains.append(d)

    return {
        "per_question": per_q,
        "domain_scores": {d: {"correct": c, "total": t} for d, (c, t) in domain_scores.items()},
        "weak_domains": weak_domains,
        "weak_topics": weak_topics
    }


if __name__ == "__main__":
    # Quick test
    quiz_path = "../data/sample_quiz.json"
    user_answers = ["4", "Oxygen", "Java", "1914"]  # fake answers

    result = evaluate_quiz(quiz_path, user_answers)
    print(json.dumps(result, indent=2))
