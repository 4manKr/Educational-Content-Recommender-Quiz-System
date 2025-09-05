# Edu- 📚

An intelligent educational platform that provides personalized learning recommendations through quiz-based assessment and machine learning-powered resource suggestions.

## 🚀 Features

- **Interactive Quiz System**: Generate and take quizzes across multiple subjects (Math, Science, CS, Physics)
- **Smart Assessment**: Evaluate quiz performance and identify weak subject areas
- **ML-Powered Recommendations**: Get personalized learning resources based on your performance
- **Comprehensive Resource Database**: Access to 2000+ curated educational resources
- **Web Interface**: User-friendly Streamlit-based frontend
- **Multiple Learning Levels**: Resources categorized by Beginner, Intermediate, and Advanced levels

## 🏗️ Project Structure

```
Edu-/
├── frontend/
│   └── app.py                 # Main Streamlit web application
├── backend/
│   ├── model_trainer.py       # ML model training for recommendations
│   ├── step6_evaluator.py     # Quiz evaluation engine
│   ├── step6_recommender.py   # Resource recommendation system
│   ├── ml_recommender.py      # ML-based recommendation engine
│   └── models/
│       └── recommender_model.pkl  # Trained ML model
├── data/
│   ├── resources_curated.csv  # Main educational resources dataset
│   ├── resources_cleaned.csv  # Processed dataset
│   ├── sample_quiz.json       # Sample quiz data
│   └── quiz_data_*.json       # Subject-specific quiz data
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Edu-
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run frontend/app.py
   ```

4. **Access the web interface**
   Open your browser and navigate to `http://localhost:8501`

## 📖 Usage

### Taking a Quiz
1. Launch the application using the command above
2. Click "Take Quiz" on the main page
3. Answer questions across different subjects
4. Submit your responses to get evaluated

### Getting Recommendations
1. After completing a quiz, the system will:
   - Analyze your performance
   - Identify weak subject areas
   - Generate personalized learning recommendations
2. Browse recommended resources with direct links

### Training the ML Model
1. Navigate to the "Train Model" section
2. Click "Train Recommendation Model" to update the ML model
3. The system will process the resource dataset and create recommendations

## 🧠 How It Works

### Quiz System
- **Question Generation**: Questions are generated from the curated resource database
- **Multi-Subject Coverage**: Covers Math, Science, Computer Science, and Physics
- **Performance Analysis**: Evaluates answers and identifies knowledge gaps

### Recommendation Engine
- **TF-IDF Vectorization**: Converts resource descriptions into numerical features
- **Nearest Neighbors**: Uses cosine similarity to find related resources
- **Personalized Suggestions**: Recommends resources based on weak subject areas

### Data Processing
- **Resource Curation**: 2000+ educational resources from various sources
- **Domain Classification**: Resources categorized by subject and difficulty level
- **Quality Assurance**: Curated dataset ensures high-quality learning materials

## 📊 Dataset

The system uses a comprehensive dataset of educational resources including:

- **Sources**: Khan Academy, MIT OpenCourseWare, OpenStax, Wikipedia
- **Subjects**: Mathematics, Science, Computer Science, History
- **Resource Types**: Videos, Courses, Articles, Textbooks
- **Difficulty Levels**: Beginner, Intermediate, Advanced

## 🔧 Configuration

### Model Parameters
- **TF-IDF Features**: 5000 maximum features
- **Nearest Neighbors**: 5 neighbors for recommendations
- **Similarity Metric**: Cosine similarity

### Quiz Settings
- **Default Questions**: 2 questions per subject
- **Passing Threshold**: 70% for subject mastery
- **Weak Subject Threshold**: 60% for identification

## 🚀 Advanced Features

### ML Model Training
```python
from backend.model_trainer import train_model

# Train with custom parameters
metrics = train_model(
    data_path="data/resources_curated.csv",
    model_path="backend/models/recommender_model.pkl",
    max_features=5000,
    n_neighbors=5
)
```

### Custom Quiz Generation
```python
from backend.quiz_engine import get_quiz

# Generate quiz with specific parameters
quiz = get_quiz(num_questions=5, domains=["Math", "Science"])
```

## 🧪 Testing

Run the demo to test the complete pipeline:
```bash
cd backend
python run_step6_demo.py
```

## 📈 Performance Metrics

The system tracks:
- **Quiz Accuracy**: Per-subject and overall performance
- **Recommendation Quality**: Resource relevance and user engagement
- **Model Performance**: Training accuracy and recommendation diversity

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Khan Academy** for educational content
- **MIT OpenCourseWare** for advanced courses
- **OpenStax** for open-source textbooks
- **Streamlit** for the web framework
- **scikit-learn** for machine learning capabilities

## 📞 Support

For support, email [your-email@example.com] or create an issue in the repository.

## 🔮 Future Enhancements

- [ ] User authentication and progress tracking
- [ ] Advanced analytics dashboard
- [ ] Mobile app development
- [ ] Integration with more educational platforms
- [ ] Adaptive learning algorithms
- [ ] Social learning features
- [ ] Gamification elements

---

**Made with ❤️ for better education**
