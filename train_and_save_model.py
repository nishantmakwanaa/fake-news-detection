#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train and Save Complete Model Pipeline for Fake News Detection

This script provides a backup way to train and save the complete model pipeline
if final_model.sav is missing or needs to be regenerated.

@author: nishant makwana

Usage:
    python train_and_save_model.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 🔧 SKLEARN COMPATIBILITY PATCH
import sys
import sklearn.linear_model._logistic
sys.modules['sklearn.linear_model.logistic'] = sklearn.linear_model._logistic


def train_and_save_model():
    """Train a new model pipeline and save it to final_model.sav"""
    
    print("=" * 60)
    print("FAKE NEWS DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    # Get base directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_PATH = os.path.join(BASE_DIR, 'train.csv')
    TEST_PATH = os.path.join(BASE_DIR, 'test.csv')
    MODEL_PATH = os.path.join(BASE_DIR, 'final_model.sav')
    
    # Check if training data exists
    if not os.path.exists(TRAIN_PATH):
        print("❌ ERROR: train.csv not found!")
        print(f"   Expected at: {TRAIN_PATH}")
        print("   Please ensure train.csv exists in the project directory.")
        return False
    
    # Load training data
    print("\n📂 Loading training data...")
    train_news = pd.read_csv(TRAIN_PATH)
    print(f"   ✅ Loaded {len(train_news)} training samples")
    
    # Check if test data exists for validation
    test_available = os.path.exists(TEST_PATH)
    if test_available:
        test_news = pd.read_csv(TEST_PATH)
        print(f"   ✅ Loaded {len(test_news)} test samples")
    else:
        print("   ⚠ test.csv not found, will split training data for validation")
        train_news, test_news = train_test_split(train_news, test_size=0.2, random_state=42)
    
    # Extract features and labels
    X_train = train_news['Statement']
    y_train = train_news['Label']
    X_test = test_news['Statement']
    y_test = test_news['Label']
    
    # Create the complete pipeline
    print("\n🔧 Creating model pipeline...")
    print("   - TF-IDF Vectorizer (ngram_range=(1,4), English stopwords)")
    print("   - Logistic Regression Classifier (L2 penalty, C=1)")
    
    pipeline = Pipeline([
        ('LogR_tfidf', TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 4),
            use_idf=True,
            smooth_idf=True
        )),
        ('LogR_clf', LogisticRegression(penalty="l2", C=1, max_iter=1000))
    ])
    
    # Train the model
    print("\n🎯 Training model...")
    pipeline.fit(X_train, y_train)
    print("   ✅ Training complete!")
    
    # Evaluate the model
    print("\n📊 Evaluating model performance...")
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n📈 Classification Report:")
    print(classification_report(y_test, predictions))
    
    # Save the complete pipeline
    print(f"\n💾 Saving model to {MODEL_PATH}...")
    pickle.dump(pipeline, open(MODEL_PATH, 'wb'))
    print(f"   ✅ Model saved successfully!")
    
    # Verify the saved model
    print("\n🔍 Verifying saved model...")
    loaded_model = pickle.load(open(MODEL_PATH, 'rb'))
    test_prediction = loaded_model.predict(["Obama is running for president"])
    print(f"   ✅ Model loaded and tested successfully!")
    print(f"   Test prediction: {test_prediction[0]}")
    
    print("\n" + "=" * 60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print("\nYou can now run the Flask application:")
    print("  python front.py")
    print("\n")
    
    return True


if __name__ == '__main__':
    success = train_and_save_model()
    if not success:
        print("\n❌ Training failed. Please check the error messages above.")
        exit(1)
