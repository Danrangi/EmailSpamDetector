import PyInstaller.__main__
import os
import nltk

# Locate NLTK data
nltk_data_path = None
for path in nltk.data.path:
    if os.path.exists(path):
        nltk_data_path = path
        break

separator = os.pathsep

PyInstaller.__main__.run([
    'app.py',
    '--name=MailGuard',
    '--onefile',
    '--noconsole',
    f'--add-data=templates{separator}templates',
    f'--add-data=static{separator}static',
    f'--add-data=models{separator}models',
    f'--add-data={nltk_data_path}{separator}nltk_data',
    '--hidden-import=sklearn.naive_bayes',
    '--hidden-import=sklearn.feature_extraction.text',
    '--hidden-import=nltk',
    '--hidden-import=joblib',
    '--clean'
])