# 🔍 Fake News Detector

An AI-powered web application that uses Machine Learning and Natural Language Processing to detect fake news articles.

**Author:** nishant makwana

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)

## 🌟 Features

- **Machine Learning Powered**: Uses TF-IDF vectorization with Logistic Regression
- **Real-time Analysis**: Get instant predictions on news authenticity
- **Modern UI**: Beautiful, responsive dark-themed interface
- **High Accuracy**: Trained on thousands of verified news articles
- **Confidence Scores**: Shows prediction confidence percentage

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd Fake_News_Detection-master
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - Mac/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install required packages**
   ```bash
   pip install flask scikit-learn pandas nltk
   ```

5. **Download NLTK data**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

### Running the Application

1. **Start the Flask server**
   ```bash
   python front.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Enter a news article** in the text box and click "Analyze Article"

## 📁 Project Structure

```
Fake_News_Detection/
├── front.py                    # Main Flask application
├── final_model.sav            # Pre-trained ML model (19MB)
├── train_and_save_model.py    # Script to retrain the model
├── prediction.py              # Standalone prediction script
├── train.csv                  # Training dataset
├── templates/                 # HTML templates
│   ├── base.html             # Base template
│   └── index.html            # Main page
├── static/                    # Static files
│   └── style.css             # Custom CSS styling
├── archive/                   # Archived files
│   ├── old_css/              # Old CSS files
│   ├── training_scripts/     # Original training scripts
│   └── datasets/             # Additional datasets
└── README.md                  # This file
```

## 🧠 How It Works

1. **Text Preprocessing**: News text is cleaned, tokenized, and lemmatized
2. **Feature Extraction**: TF-IDF vectorization with n-grams (1-4)
3. **Classification**: Logistic Regression model predicts fake vs real
4. **Result Display**: Shows prediction with confidence score

## 🔧 Retraining the Model

If you want to retrain the model with your own data:

```bash
python train_and_save_model.py
```

This will:
- Load training data from `train.csv`
- Train a new model pipeline
- Save it as `final_model.sav`
- Display accuracy metrics

## 🎨 UI Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Dark Theme**: Easy on the eyes with gradient backgrounds
- **Smooth Animations**: Professional transitions and effects
- **Accessible**: Keyboard navigation and screen reader friendly

## 📊 Model Details

- **Algorithm**: Logistic Regression
- **Vectorization**: TF-IDF with n-grams (1-4)
- **Features**: English stopwords removal, lemmatization
- **Training Data**: Verified news articles dataset

## 🛠️ Technologies Used

- **Backend**: Flask (Python web framework)
- **ML Libraries**: scikit-learn, NLTK
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Styling**: Custom CSS with modern design patterns

## 📝 Usage Example

```python
# Standalone prediction
from prediction import detecting_fake_news

news_text = "Breaking: Major political announcement today..."
detecting_fake_news(news_text)
```

## 🤝 Contributing

This is an educational project. Feel free to:
- Report bugs
- Suggest new features
- Improve the model accuracy
- Enhance the UI/UX

## 📄 License

See LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: LIAR dataset for fake news detection
- Libraries: scikit-learn, NLTK, Flask
- Inspiration: Fighting misinformation through AI

## 📞 Support

If you encounter any issues:
1. Make sure all dependencies are installed
2. Check that `final_model.sav` exists (19MB file)
3. Verify Python version is 3.8+
4. Ensure NLTK data is downloaded

## 🎯 Future Improvements

- [ ] Add support for multiple languages
- [ ] Implement source credibility checking
- [ ] Add fact-checking API integration
- [ ] Enhance model with deep learning
- [ ] Add user authentication and history

---

**Author:** nishant makwana

**Made with ❤️ using Python & Machine Learning**
