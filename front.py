# -*- coding: utf-8 -*-
"""
Fake News Detection Web Application

@author: nishant makwana
"""

from flask import Flask, render_template, request
import warnings
warnings.filterwarnings("ignore")

# 🔧 SKLEARN COMPATIBILITY PATCH (VERY IMPORTANT)
# This fixes compatibility issues with older sklearn models
import sys
import sklearn.linear_model._logistic
sys.modules['sklearn.linear_model.logistic'] = sklearn.linear_model._logistic

import re
import nltk
import os
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Initialize NLTK data - fix WordNet error
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Fix WordNet corpus loader issue - handle the _LazyCorpusLoader__args error
try:
    from nltk.corpus import wordnet
    # Force initialization of wordnet by accessing it
    _ = wordnet.synsets('test')
except (LookupError, AttributeError, Exception) as e:
    print(f"Downloading wordnet corpus...")
    nltk.download('wordnet', quiet=True)
    # Re-import after download
    import importlib
    import nltk.corpus
    if 'wordnet' in sys.modules:
        del sys.modules['nltk.corpus.wordnet']
    from nltk.corpus import wordnet
    # Force initialization
    try:
        _ = wordnet.synsets('test')
    except:
        pass

# Initialize lemmatizer and stopwords with error handling
try:
    lemmatizer = WordNetLemmatizer()
    stpwrds = set(stopwords.words('english'))
except Exception as e:
    print(f"Warning: Error initializing NLTK components: {e}")
    print("Attempting to re-download required NLTK data...")
    # Fallback initialization
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        lemmatizer = WordNetLemmatizer()
        stpwrds = set(stopwords.words('english'))
    except Exception as e2:
        print(f"Critical error: Could not initialize NLTK components: {e2}")
        raise

# Load the complete trained pipeline (includes fitted TF-IDF vectorizer + classifier)
# Use absolute path based on script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'final_model.sav')
loaded_pipeline = None

def load_model():
    """Load the model from local file or external source if needed."""
    global loaded_pipeline
    
    # Try loading from local file first
    if os.path.exists(MODEL_PATH):
        try:
            loaded_pipeline = pickle.load(open(MODEL_PATH, 'rb'))
            print(f"✅ Model loaded successfully from {MODEL_PATH}")
            print(f"   Model type: {type(loaded_pipeline)}")
            return True
        except Exception as e:
            print(f"❌ ERROR loading model from local file: {str(e)}")
    
    # If local file doesn't exist, try loading from external source
    # You can set MODEL_URL environment variable in Vercel to point to your model
    # Example: https://github.com/yourusername/repo/releases/download/v1.0/final_model.sav
    model_url = os.environ.get('MODEL_URL')
    if model_url:
        try:
            import urllib.request
            print(f"📥 Downloading model from external source: {model_url}")
            urllib.request.urlretrieve(model_url, MODEL_PATH)
            loaded_pipeline = pickle.load(open(MODEL_PATH, 'rb'))
            print(f"✅ Model loaded successfully from external source")
            return True
        except Exception as e:
            print(f"❌ ERROR loading model from external source: {str(e)}")
    
    print(f"❌ ERROR: Model file '{MODEL_PATH}' not found!")
    print("   Please ensure final_model.sav exists or set MODEL_URL environment variable.")
    loaded_pipeline = None
    return False

# Load model on startup
load_model()


def fake_news_det(news):
    """
    Detect if news is fake or real using the trained pipeline.
    Returns a dictionary with all prediction details.
    """
    if loaded_pipeline is None:
        return {
            'error': "Model not loaded. Please check if final_model.sav exists.",
            'statement': news,
            'prediction': None,
            'prediction_value': None,
            'prediction_text': None,
            'real_probability': None,
            'fake_probability': None,
            'confidence': None
        }
    
    try:
        # The pipeline expects raw text - TF-IDF vectorizer handles preprocessing internally
        # Just pass the original text directly to match how the model was trained
        # (The pipeline's TF-IDF has stop_words='english' built-in)
        prediction = loaded_pipeline.predict([news])
        
        # Get prediction probability for confidence score
        try:
            prob = loaded_pipeline.predict_proba([news])
            # prob[0][0] is probability of class 0 (Fake), prob[0][1] is probability of class 1 (Real)
            fake_prob = prob[0][0]
            real_prob = prob[0][1]
            
            # Map prediction: 0 = Fake, 1 = Real
            prediction_value = int(prediction[0])
            is_real = prediction_value == 1
            prediction_text = "Real" if is_real else "Fake"
            
            return {
                'error': None,
                'statement': news,
                'prediction': is_real,
                'prediction_value': prediction_value,
                'prediction_text': prediction_text,
                'real_probability': float(real_prob),
                'fake_probability': float(fake_prob),
                'confidence': float(real_prob if is_real else fake_prob)
            }
        except Exception as e:
            # Fallback if predict_proba is not available
            prediction_value = int(prediction[0])
            is_real = prediction_value == 1
            prediction_text = "Real" if is_real else "Fake"
            return {
                'error': None,
                'statement': news,
                'prediction': is_real,
                'prediction_value': prediction_value,
                'prediction_text': prediction_text,
                'real_probability': None,
                'fake_probability': None,
                'confidence': None
            }
            
    except Exception as e:
        return {
            'error': str(e),
            'statement': news,
            'prediction': None,
            'prediction_value': None,
            'prediction_text': None,
            'real_probability': None,
            'fake_probability': None,
            'confidence': None
        }


@app.route('/')
def home():
    return render_template('index.html', prediction_result=None)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['news']
        if not message or message.strip() == '':
            return render_template('index.html', prediction_result={
                'error': "Please enter some news text to analyze.",
                'statement': '',
                'prediction': None,
                'prediction_value': None,
                'prediction_text': None,
                'real_probability': None,
                'fake_probability': None,
                'confidence': None
            })
        
        pred_result = fake_news_det(message)
        return render_template('index.html', prediction_result=pred_result)
    else:
        return render_template('index.html', prediction_result={
            'error': "Something went wrong",
            'statement': '',
            'prediction': None,
            'prediction_value': None,
            'prediction_text': None,
            'real_probability': None,
            'fake_probability': None,
            'confidence': None
        })


if __name__ == '__main__':
    app.run(debug=True)