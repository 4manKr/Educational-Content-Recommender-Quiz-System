import requests, json

domains = {
    "math": "https://opentdb.com/api.php?amount=10&category=19&type=multiple",
    "science": "https://opentdb.com/api.php?amount=10&category=17&type=multiple",
    "programming": "https://opentdb.com/api.php?amount=10&category=18&type=multiple",
    "history": "https://opentdb.com/api.php?amount=10&category=23&type=multiple",
}

for subject, url in domains.items():
    res = requests.get(url).json()
    with open(f"data/quiz_data_{subject}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved {subject} quiz in data/quiz_data_{subject}.json")
