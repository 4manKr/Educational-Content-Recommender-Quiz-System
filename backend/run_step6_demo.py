import json
from step6_evaluator import evaluate_quiz
from step6_recommender import recommend_from_weakness


if __name__ == "__main__":
    # Load quiz file
    quiz_path = "../data/sample_quiz.json"

    # Fake user answers (for demo) – you’ll replace with actual quiz input later
    user_answers = ["4", "Oxygen", "Python", "1939"]

    # Step 1: Evaluate quiz
    summary = evaluate_quiz(quiz_path, user_answers)

    print("\n🔹 Domain Scores:")
    print(json.dumps(summary["domain_scores"], indent=2))

    print("\n🔹 Weak Domains:")
    print(summary["weak_domains"])

    print("\n🔹 Weak Topics:")
    print(summary["weak_topics"])

    # Step 2: Get recommendations
    recs = recommend_from_weakness(summary["weak_domains"], summary["weak_topics"])

    print("\n✅ Recommended Resources:")
    for key, items in recs.items():
        print(f"\n{key}")
        for r in items:
            print(f"- {r['title']} ({r['type']} - {r['level']}) → {r['url']}")
