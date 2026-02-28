"""
app.py
------
Flask web application that serves the spam classifier.
Users can type or paste email content and get instant predictions.
"""

import os
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request, jsonify

# Ensure NLTK data is available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize NLP tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
MODEL_PATH = os.path.join('models', 'spam_classifier_model.pkl')
VECTORIZER_PATH = os.path.join('models', 'tfidf_vectorizer.pkl')

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print(f"✓ Model loaded from {MODEL_PATH}")
    print(f"✓ Vectorizer loaded from {VECTORIZER_PATH}")
except FileNotFoundError:
    print("ERROR: Model files not found!")
    print("Run 'python preprocess.py' then 'python train_model.py' first.")
    exit(1)


def clean_text(text):
    """
    Same preprocessing pipeline used during training.
    Must be identical to ensure consistent predictions.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    cleaned_tokens = [
        stemmer.stem(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    return ' '.join(cleaned_tokens)


def predict_email(email_text):
    """
    Takes raw email text, cleans it, vectorizes it,
    and returns the prediction with confidence scores.
    """
    # Clean the text
    cleaned = clean_text(email_text)
    
    if not cleaned.strip():
        return {
            'prediction': 'unknown',
            'confidence': 0,
            'cleaned_text': '',
            'message': 'Could not extract meaningful text from input.'
        }
    
    # Vectorize
    text_vector = vectorizer.transform([cleaned])
    
    # Predict
    prediction = model.predict(text_vector)[0]
    
    # Get probability scores
    probabilities = model.predict_proba(text_vector)[0]
    classes = model.classes_
    
    # Build confidence dictionary
    confidence_dict = {}
    for cls, prob in zip(classes, probabilities):
        confidence_dict[cls] = round(prob * 100, 2)
    
    return {
        'prediction': prediction,
        'confidence': confidence_dict.get(prediction, 0),
        'ham_confidence': confidence_dict.get('ham', 0),
        'spam_confidence': confidence_dict.get('spam', 0),
        'cleaned_text': cleaned
    }


@app.route('/')
def home():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    email_text = request.form.get('email_text', '').strip()
    
    if not email_text:
        return render_template('index.html', error='Please enter some email text.')
    
    result = predict_email(email_text)
    
    return render_template(
        'index.html',
        prediction=result['prediction'],
        confidence=result['confidence'],
        ham_confidence=result['ham_confidence'],
        spam_confidence=result['spam_confidence'],
        email_text=email_text,
        cleaned_text=result['cleaned_text']
    )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access."""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Please provide "text" field in JSON body.'}), 400
    
    result = predict_email(data['text'])
    return jsonify(result)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  SPAM CLASSIFIER WEB APPLICATION")
    print("  Open your browser to: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)