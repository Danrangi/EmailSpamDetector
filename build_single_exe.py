import PyInstaller.__main__
import os
import nltk

# Locate NLTK data
nltk_data_path = None
for path in nltk.data.path:
    if os.path.exists(path):
        nltk_data_path = path
        break

PyInstaller.__main__.run([
    'app.py',
    '--name=MailGuard',
    '--onefile',
    '--noconsole',
    '--add-data=templates:templates',
    '--add-data=static:static',
    '--add-data=models:models',
    f'--add-data={nltk_data_path}:nltk_data',
    '--hidden-import=sklearn.naive_bayes',
    '--hidden-import=sklearn.feature_extraction.text',
    '--hidden-import=nltk',
    '--hidden-import=joblib',
    '--clean'
])