"""
preprocess.py
--------------
Handles loading the Enron dataset (folder-based or CSV-based)
and cleaning/preprocessing the text data.
Outputs a cleaned CSV file into the data/ folder.
"""

import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Make sure NLTK data is available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    """
    Takes raw email text and applies:
    1. Lowercase conversion (case folding)
    2. Remove non-alphabetic characters
    3. Tokenization
    4. Stop-word removal
    5. Stemming (Porter Stemmer)
    
    Returns the cleaned string.
    """
    # Step 1: Case folding
    text = text.lower()
    
    # Step 2: Remove non-alphabetic characters (keep spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Step 3: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Step 4: Tokenization
    tokens = word_tokenize(text)
    
    # Step 5: Stop-word removal and Stemming
    cleaned_tokens = [
        stemmer.stem(word) 
        for word in tokens 
        if word not in stop_words and len(word) > 2
    ]
    
    return ' '.join(cleaned_tokens)


def load_from_folders(data_path):
    """
    Loads emails from Enron folder structure:
    data_path/
        enron1/
            ham/
                file1.txt
                file2.txt
            spam/
                file1.txt
                file2.txt
        enron2/
            ...
    
    Also handles a simpler structure:
    data_path/
        ham/
            file1.txt
        spam/
            file1.txt
    
    Returns a DataFrame with columns: ['text', 'label']
    """
    emails = []
    labels = []
    
    # Check if data_path directly contains ham/ and spam/
    direct_ham = os.path.join(data_path, 'ham')
    direct_spam = os.path.join(data_path, 'spam')
    
    if os.path.isdir(direct_ham) or os.path.isdir(direct_spam):
        # Simple structure: data/ham/ and data/spam/
        folders_to_process = [data_path]
    else:
        # Nested structure: data/enron1/, data/enron2/, etc.
        folders_to_process = [
            os.path.join(data_path, folder) 
            for folder in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, folder))
        ]
    
    for folder in folders_to_process:
        for label in ['ham', 'spam']:
            label_path = os.path.join(folder, label)
            if not os.path.isdir(label_path):
                continue
                
            file_list = os.listdir(label_path)
            print(f"  Loading {len(file_list)} {label} emails from {label_path}")
            
            for filename in file_list:
                filepath = os.path.join(label_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    emails.append(content)
                    labels.append(label)
                except Exception as e:
                    print(f"  Skipping {filepath}: {e}")
                    continue
    
    df = pd.DataFrame({'text': emails, 'label': labels})
    return df


def load_from_csv(csv_path):
    """
    Loads emails from a CSV file.
    Tries to detect common column name patterns.
    
    Returns a DataFrame with columns: ['text', 'label']
    """
    df = pd.read_csv(csv_path, encoding='utf-8', errors='ignore')
    
    print(f"  CSV columns found: {list(df.columns)}")
    print(f"  CSV shape: {df.shape}")
    
    # Try to identify the text and label columns
    # Common patterns in Enron/spam datasets
    text_col = None
    label_col = None
    
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ['text', 'message', 'email', 'body', 'content', 'v2']:
            text_col = col
        if col_lower in ['label', 'class', 'category', 'spam', 'type', 'v1', 'target']:
            label_col = col
    
    # If we couldn't find them automatically, try positional
    if text_col is None or label_col is None:
        if len(df.columns) >= 2:
            # Assume first column is label, second is text (common format)
            label_col = df.columns[0]
            text_col = df.columns[1]
            print(f"  Auto-detected: label column = '{label_col}', text column = '{text_col}'")
    
    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not identify text and label columns. "
            f"Columns found: {list(df.columns)}. "
            f"Please rename them to 'text' and 'label'."
        )
    
    result = pd.DataFrame({
        'text': df[text_col].astype(str),
        'label': df[label_col].astype(str).str.lower().str.strip()
    })
    
    # Standardize labels
    label_mapping = {
        '1': 'spam', '0': 'ham',
        'spam': 'spam', 'ham': 'ham',
        'yes': 'spam', 'no': 'ham',
        'true': 'spam', 'false': 'ham'
    }
    result['label'] = result['label'].map(label_mapping).fillna(result['label'])
    
    return result


def main():
    """
    Main preprocessing pipeline.
    Detects whether the data is folder-based or CSV-based,
    loads it, cleans it, and saves the result.
    """
    data_path = 'data'
    output_path = os.path.join('data', 'cleaned_data.csv')
    
    print("=" * 60)
    print("STEP 1: LOADING DATA")
    print("=" * 60)
    
    # Check what's in the data folder
    data_contents = os.listdir(data_path)
    csv_files = [f for f in data_contents if f.endswith('.csv') and f != 'cleaned_data.csv']
    
    if csv_files:
        # Load from CSV
        csv_path = os.path.join(data_path, csv_files[0])
        print(f"Found CSV file: {csv_path}")
        df = load_from_csv(csv_path)
    else:
        # Load from folders
        print("No CSV found. Looking for folder structure (ham/spam folders)...")
        df = load_from_folders(data_path)
    
    if df.empty:
        print("\nERROR: No data was loaded!")
        print("Make sure your data is in the 'data/' folder as either:")
        print("  - A CSV file with 'text' and 'label' columns")
        print("  - Folders with 'ham/' and 'spam/' subfolders containing .txt files")
        return
    
    print(f"\nTotal emails loaded: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Drop any rows with missing text
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip() != '']
    print(f"After removing empty rows: {len(df)}")
    
    print("\n" + "=" * 60)
    print("STEP 2: CLEANING TEXT")
    print("=" * 60)
    
    total = len(df)
    cleaned_texts = []
    
    for i, text in enumerate(df['text']):
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"  Processing email {i + 1}/{total}...")
        cleaned_texts.append(clean_text(text))
    
    df['cleaned_text'] = cleaned_texts
    
    # Remove rows where cleaning produced empty strings
    df = df[df['cleaned_text'].str.strip() != '']
    print(f"After cleaning: {len(df)} emails remain")
    
    # Show some examples
    print("\n--- Sample Cleaned Data ---")
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        print(f"\n[{row['label'].upper()}]")
        print(f"  Original : {row['text'][:100]}...")
        print(f"  Cleaned  : {row['cleaned_text'][:100]}...")
    
    print("\n" + "=" * 60)
    print("STEP 3: SAVING CLEANED DATA")
    print("=" * 60)
    
    df[['cleaned_text', 'label']].to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")
    print(f"Final dataset size: {len(df)}")
    print(f"Final label distribution:\n{df['label'].value_counts()}")
    
    print("\n✓ Preprocessing complete! Run 'python train_model.py' next.")


if __name__ == '__main__':
    main()