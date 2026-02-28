import os
import sys
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'spam_classifier_model.pkl')
VEC_PATH = os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl')

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
except Exception as e:
    print(f"Error loading models: {e}")
    sys.exit(1)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    cleaned = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(cleaned)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    email_text = ""
    confidence = 0
    ham_confidence = 0
    spam_confidence = 0
    cleaned_text = ""
    
    if request.method == 'POST':
        email_text = request.form.get('email_text', '')
        if email_text.strip():
            cleaned_text = clean_text(email_text)
            if cleaned_text:
                vec_text = vectorizer.transform([cleaned_text])
                pred = model.predict(vec_text)[0]
                
                prob = model.predict_proba(vec_text)[0]
                spam_confidence = round(prob[0] * 100, 2)
                ham_confidence = round(prob[1] * 100, 2)
                
                confidence = ham_confidence if pred == 1 else spam_confidence
                prediction = 'ham' if pred == 1 else 'spam'
            else:
                prediction = 'unknown'
                
    return render_template('index.html', 
                           prediction=prediction, 
                           email_text=email_text,
                           confidence=confidence,
                           ham_confidence=ham_confidence,
                           spam_confidence=spam_confidence,
                           cleaned_text=cleaned_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)