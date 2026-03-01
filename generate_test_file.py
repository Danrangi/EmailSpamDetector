import pandas as pd
import random
import os

def generate_test_file():
    # Define paths
    csv_path = os.path.join('data', 'mail_data.csv')
    output_path = 'presentation_test_emails.txt'
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    # Load data using encoding_errors
    df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='ignore')
    
    # Standardize column names
    if 'Category' in df.columns:
        df.rename(columns={'Category': 'label', 'Message': 'text'}, inplace=True)
        
    # Filter emails by category
    spam_emails = df[df['label'].str.lower() == 'spam']['text'].dropna().tolist()
    ham_emails = df[df['label'].str.lower() == 'ham']['text'].dropna().tolist()
    
    # Select 50 of each
    sample_spam = random.sample(spam_emails, min(50, len(spam_emails)))
    sample_ham = random.sample(ham_emails, min(50, len(ham_emails)))
    
    # Write to a formatted text file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=== SPAM EMAILS FOR PRESENTATION ===\n")
        f.write("Copy the text below the dashed lines to test the spam detection.\n\n")
        
        for i, email in enumerate(sample_spam, 1):
            f.write(f"--- Spam Email #{i} ---\n{email}\n\n")
            
        f.write("\n=== LEGITIMATE (HAM) EMAILS FOR PRESENTATION ===\n")
        f.write("Copy the text below the dashed lines to test legitimate email detection.\n\n")
        
        for i, email in enumerate(sample_ham, 1):
            f.write(f"--- Ham Email #{i} ---\n{email}\n\n")

    print(f"Success! Open '{output_path}' to see your 100 test emails.")

if __name__ == '__main__':
    generate_test_file()