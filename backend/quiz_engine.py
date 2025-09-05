import random

# Sample quiz bank (you can expand this easily)
QUIZ_BANK = {
    "Math": [
        {"q": "What is the derivative of x²?", "options": ["2x", "x", "x²", "1"], "answer": "2x"},
        {"q": "Solve for x: 2x + 3 = 7", "options": ["2", "1", "3", "4"], "answer": "2"},
    ],
    "Physics": [
        {"q": "Who formulated the laws of motion?", "options": ["Newton", "Einstein", "Galileo", "Tesla"], "answer": "Newton"},
        {"q": "What is the unit of Force?", "options": ["Newton", "Joule", "Pascal", "Watt"], "answer": "Newton"},
    ],
    "CS": [
        {"q": "Python is what type of language?", "options": ["Compiled", "Interpreted", "Assembly", "Machine"], "answer": "Interpreted"},
        {"q": "Which data structure uses FIFO?", "options": ["Stack", "Queue", "Tree", "Graph"], "answer": "Queue"},
    ],
}

def get_quiz(num_questions=2):
    """Pick random questions from each subject"""
    quiz = {}
    for subject, questions in QUIZ_BANK.items():
        quiz[subject] = random.sample(questions, min(num_questions, len(questions)))
    return quiz

def evaluate_quiz(responses):
    """Evaluate quiz and return weak subjects"""
    weak_subjects = []
    for subject, answers in responses.items():
        correct = 0
        for i, user_ans in enumerate(answers):
            if user_ans == QUIZ_BANK[subject][i]["answer"]:
                correct += 1
        score = correct / len(answers)
        if score < 0.7:  # threshold → less than 70% is weak
            weak_subjects.append(subject)
    return weak_subjects

