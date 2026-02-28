import os
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
df = pd.read_csv(url, encoding='latin-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'spam': 0, 'ham': 1})
df = df.dropna()

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    cleaned = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(cleaned)

print("Cleaning text and training model...")
df['cleaned_text'] = df['text'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Model Trained! Accuracy: {acc*100:.2f}%")

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/spam_classifier_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
print("Model and vectorizer saved to models/ folder.")