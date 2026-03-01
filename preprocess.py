import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    cleaned = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(cleaned)

def main():
    os.makedirs('data', exist_ok=True)
    csv_path = os.path.join('data', 'mail_data.csv')
    output_path = os.path.join('data', 'cleaned_data.csv')

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    # FIXED: Using encoding_errors instead of errors
    df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='ignore')
    
    if 'Category' in df.columns:
        df.rename(columns={'Category': 'label', 'Message': 'text'}, inplace=True)
    elif 'v1' in df.columns:
        df.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)

    df['label'] = df['label'].map({'spam': 'spam', 'ham': 'ham', 0: 'spam', 1: 'ham'}).fillna('ham')
    df = df.dropna(subset=['text'])
    
    print("Cleaning text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    df = df[df['cleaned_text'].str.strip() != '']
    
    df[['cleaned_text', 'label']].to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

if __name__ == '__main__':
    main()