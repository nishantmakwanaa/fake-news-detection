# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:45:40 2017

@author: nishant makwana


"""

import warnings
warnings.filterwarnings("ignore")


# 🔧 COMPATIBILITY PATCH (VERY IMPORTANT)
import sys
import sklearn.linear_model._logistic
sys.modules['sklearn.linear_model.logistic'] = sklearn.linear_model._logistic

import pickle
import os

var = input("Please enter the news text you want to verify: ")
print("You entered:", var)


# function to run for prediction
def detecting_fake_news(var):
    # Get the base directory where the script is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'final_model.sav')
    
    # load trained model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model file '{MODEL_PATH}' not found!")
        return
    
    load_model = pickle.load(open(MODEL_PATH, 'rb'))

    # The pipeline expects raw text - TF-IDF handles preprocessing internally
    prediction = load_model.predict([var])
    prob = load_model.predict_proba([var])

    # Map prediction: 0 = Fake, 1 = Real
    prediction_value = int(prediction[0])
    is_real = prediction_value == 1
    prediction_label = "Real" if is_real else "Fake"
    
    # Get probabilities
    fake_prob = prob[0][0]
    real_prob = prob[0][1]
    confidence = real_prob if is_real else fake_prob

    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"\n📰 Statement Analyzed:")
    print(f"   {var}")
    print(f"\n🔍 Prediction Result:")
    print(f"   The given statement is: {prediction_label}")
    print(f"\n📊 Probability Scores:")
    print(f"   Real News Probability: {real_prob:.4f} ({real_prob*100:.2f}%)")
    print(f"   Fake News Probability: {fake_prob:.4f} ({fake_prob*100:.2f}%)")
    print(f"\n💯 Confidence Score: {confidence:.4f} ({confidence*100:.2f}%)")
    print("="*60)


if __name__ == '__main__':
    detecting_fake_news(var)
