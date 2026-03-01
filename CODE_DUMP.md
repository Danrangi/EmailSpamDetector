# EmailSpamDetector - Complete Code Dump
Generated: March 1, 2026

---

## Table of Contents
1. [app.py](#apppy) - Flask Web Application
2. [preprocess.py](#preprocesspy) - Data Preprocessing
3. [train_model.py](#train_modelpy) - Model Training
4. [build_single_exe.py](#build_single_exepy) - Build Script
5. [generate_test_file.py](#generate_test_filepy) - Test Data Generator
6. [static/main.js](#staticmainjs) - Frontend JavaScript
7. [static/style.css](#staticstylecss) - Frontend Styling
8. [templates/index.html](#templatesindexhtml) - HTML Template
9. [requirements.txt](#requirementstxt) - Dependencies
10. [pyproject.toml](#pyprojecttoml) - Project Configuration
11. [MailGuard.spec](#mailguardspec) - PyInstaller Spec

---

## app.py

```python
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
```

---

## preprocess.py

```python
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
```

---

## train_model.py

```python
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
```

---

## build_single_exe.py

```python
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
```

---

## generate_test_file.py

```python
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
```

---

## static/main.js

```javascript
document.addEventListener('DOMContentLoaded', () => {
    // 1. Fetch backend data from the hidden HTML div
    const appData = document.getElementById('app-data');
    const serverPrediction = appData.dataset.prediction;
    const serverEmailText = appData.dataset.emailText;
    const serverConfidence = parseFloat(appData.dataset.confidence || 0);

    // 2. Setup variables
    const avatarColors = ['#2563eb','#d97706','#7c3aed','#059669','#dc2626','#0891b2','#c026d3','#ea580c'];
    let inboxMessages = [];
    let spamMessages = [];
    let activeTab = 'compose';
    let selectedId = null;
    let idCounter = 1;

    // 3. Helper Functions
    function getInitials(text) {
        return text.substring(0, 2).toUpperCase();
    }

    function getColor(id) {
        return avatarColors[id % avatarColors.length];
    }

    function now() {
        const d = new Date();
        let h = d.getHours(), m = d.getMinutes();
        const ampm = h >= 12 ? 'PM' : 'AM';
        h = h % 12 || 12;
        return h + ':' + (m < 10 ? '0' : '') + m + ' ' + ampm;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // 4. Initial processing from Server Prediction
    if (serverPrediction && serverPrediction !== 'unknown') {
        const firstLine = serverEmailText.split('\n')[0].substring(0, 60) || 'No subject';
        const preview = serverEmailText.substring(0, 120);
        const msg = {
            id: idCounter++,
            sender: serverPrediction === 'spam' ? 'Suspicious Sender' : 'Unknown Sender',
            subject: firstLine,
            preview: preview,
            body: serverEmailText,
            time: now(),
            read: false,
            confidence: serverConfidence,
            prediction: serverPrediction
        };
        if (serverPrediction === 'spam') {
            spamMessages.unshift(msg);
        } else {
            inboxMessages.unshift(msg);
        }
    }

    // 5. Local Storage Management
    function loadMessages() {
        try {
            const saved = localStorage.getItem('mailguard_inbox');
            if (saved) inboxMessages = [...JSON.parse(saved), ...inboxMessages];
            const savedSpam = localStorage.getItem('mailguard_spam');
            if (savedSpam) spamMessages = [...JSON.parse(savedSpam), ...spamMessages];
            
            // Deduplicate by body
            const seenInbox = new Set();
            inboxMessages = inboxMessages.filter(m => {
                if (seenInbox.has(m.body)) return false;
                seenInbox.add(m.body); return true;
            });
            
            const seenSpam = new Set();
            spamMessages = spamMessages.filter(m => {
                if (seenSpam.has(m.body)) return false;
                seenSpam.add(m.body); return true;
            });
            
            let c = 1;
            inboxMessages.forEach(m => m.id = c++);
            spamMessages.forEach(m => m.id = c++);
            idCounter = c;
        } catch(e) {}
    }

    function saveMessages() {
        try {
            localStorage.setItem('mailguard_inbox', JSON.stringify(inboxMessages));
            localStorage.setItem('mailguard_spam', JSON.stringify(spamMessages));
        } catch(e) {}
    }

    function updateCounts() {
        document.getElementById('inbox-count').textContent = inboxMessages.length;
        document.getElementById('spam-count').textContent = spamMessages.length;
    }

    // 6. UI Rendering
    window.switchTab = function(tab) {
        activeTab = tab;
        selectedId = null;

        document.getElementById('nav-inbox').classList.toggle('active', tab === 'inbox');
        document.getElementById('nav-spam').classList.toggle('active', tab === 'spam');
        document.getElementById('nav-compose').classList.toggle('active', tab === 'compose');

        const composeView = document.getElementById('compose-view');
        const detailView = document.getElementById('email-detail-view');
        const panelTitle = document.getElementById('panel-title');
        const panelSub = document.getElementById('panel-sub');

        if (tab === 'compose') {
            composeView.style.display = 'block';
            detailView.style.display = 'none';
            panelTitle.textContent = 'History';
            const all = [...inboxMessages, ...spamMessages];
            panelSub.textContent = all.length + ' total message' + (all.length !== 1 ? 's' : '');
            renderList(all.sort((a,b) => b.id - a.id));
        } else if (tab === 'inbox') {
            composeView.style.display = 'none';
            detailView.style.display = 'none';
            panelTitle.textContent = 'Inbox';
            panelSub.textContent = inboxMessages.length + ' message' + (inboxMessages.length !== 1 ? 's' : '');
            renderList(inboxMessages);
            showPlaceholder();
        } else {
            composeView.style.display = 'none';
            detailView.style.display = 'none';
            panelTitle.textContent = 'Spam';
            panelSub.textContent = spamMessages.length + ' message' + (spamMessages.length !== 1 ? 's' : '') + (spamMessages.length > 0 ? ' — be careful' : '');
            renderList(spamMessages);
            showPlaceholder();
        }
    };

    function showPlaceholder() {
        const dv = document.getElementById('email-detail-view');
        dv.style.display = 'flex';
        dv.innerHTML = `
            <div class="detail-placeholder">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="2" y="4" width="20" height="16" rx="2"/>
                    <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>
                </svg>
                <p>Select a message to read</p>
                <span>${activeTab === 'inbox' ? inboxMessages.length + ' conversations in your inbox' : spamMessages.length + ' messages flagged as spam'}</span>
            </div>
        `;
    }

    function renderList(messages) {
        const listEl = document.getElementById('message-list');
        const clearBtn = document.getElementById('clear-list-btn');
        
        if (messages.length > 0) {
            clearBtn.style.display = 'block';
        } else {
            clearBtn.style.display = 'none';
        }

        if (messages.length === 0) {
            listEl.innerHTML = `
                <div class="empty-state">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="2" y="4" width="20" height="16" rx="2"/>
                        <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>
                    </svg>
                    <p>Nothing here</p>
                    <span>${activeTab === 'inbox' ? "You're all clear" : activeTab === 'spam' ? 'No spam detected' : 'Classify emails to see them here'}</span>
                </div>
            `;
            return;
        }

        listEl.innerHTML = messages.map(msg => {
            const isSpam = msg.prediction === 'spam';
            const color = isSpam ? '#dc2626' : getColor(msg.id);
            const initials = getInitials(msg.sender);
            return `
                <div class="message-item ${selectedId === msg.id ? 'selected' : ''}" onclick="selectMessage(${msg.id})">
                    ${!msg.read ? '<div class="msg-unread-dot"></div>' : ''}
                    <div class="msg-avatar" style="background:${color}">${initials}</div>
                    <div class="msg-content">
                        <div class="msg-top">
                            <span class="msg-sender ${msg.read ? 'read' : ''}">${escapeHtml(msg.sender)}</span>
                            <span class="msg-time">${msg.time}</span>
                        </div>
                        <div class="msg-subject">${escapeHtml(msg.subject)}</div>
                        <div class="msg-preview">${escapeHtml(msg.preview)}</div>
                    </div>
                    <button onclick="event.stopPropagation(); deleteMessage(${msg.id})" style="background:transparent; border:none; color:#64748b; cursor:pointer; padding:4px; display:flex; align-items:center;">
                        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polyline points="3 6 5 6 21 6"></polyline>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        </svg>
                    </button>
                </div>
            `;
        }).join('');
    }

    // 7. Actions
    window.selectMessage = function(id) {
        selectedId = id;
        const allMsgs = [...inboxMessages, ...spamMessages];
        const msg = allMsgs.find(m => m.id === id);
        if (!msg) return;

        msg.read = true;
        saveMessages();

        const isSpam = msg.prediction === 'spam';
        const color = isSpam ? '#dc2626' : getColor(msg.id);

        document.getElementById('compose-view').style.display = 'none';
        const dv = document.getElementById('email-detail-view');
        dv.style.display = 'flex';

        const warningHtml = isSpam ? `
            <div class="spam-warning">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
                    <path d="M12 9v4"/><path d="M12 17h.01"/>
                </svg>
                <p>This message was identified as spam. Be cautious with any links or attachments.</p>
            </div>
        ` : '';

        const moveBtn = isSpam
            ? `<button class="btn-action move-inbox" onclick="moveToInbox(${id})">Not Spam</button>`
            : `<button class="btn-action move-spam" onclick="moveToSpam(${id})">Report Spam</button>`;

        dv.innerHTML = `
            <div class="detail-email-header-wrap">
                <div style="flex:1; min-width:0;">
                    <div class="detail-email-subject">${escapeHtml(msg.subject)}</div>
                    <div class="detail-email-meta">
                        <div class="msg-avatar" style="background:${color};width:28px;height:28px;font-size:10px;">${getInitials(msg.sender)}</div>
                        <span class="meta-sender">${escapeHtml(msg.sender)}</span>
                        <span class="meta-time">· ${msg.time}</span>
                        <span class="meta-badge" style="background:${isSpam ? 'rgba(239, 68, 68, 0.1)' : 'rgba(34, 197, 94, 0.1)'};color:${isSpam ? '#f87171' : '#4ade80'};">${isSpam ? 'Spam' : 'Safe'} · ${msg.confidence}%</span>
                    </div>
                </div>
                <div class="detail-actions">
                    ${moveBtn}
                    <button class="btn-action delete" onclick="deleteMessage(${id})">Delete</button>
                </div>
            </div>
            <div class="detail-email">
                ${warningHtml}
                <div class="detail-email-body">${escapeHtml(msg.body)}</div>
            </div>
        `;

        if (activeTab === 'inbox') renderList(inboxMessages);
        else if (activeTab === 'spam') renderList(spamMessages);
        else renderList([...inboxMessages, ...spamMessages].sort((a,b) => b.id - a.id));
    };

    window.moveToSpam = function(id) {
        const idx = inboxMessages.findIndex(m => m.id === id);
        if (idx > -1) {
            const msg = inboxMessages.splice(idx, 1)[0];
            msg.prediction = 'spam';
            spamMessages.unshift(msg);
            finalizeAction();
        }
    };

    window.moveToInbox = function(id) {
        const idx = spamMessages.findIndex(m => m.id === id);
        if (idx > -1) {
            const msg = spamMessages.splice(idx, 1)[0];
            msg.prediction = 'ham';
            inboxMessages.unshift(msg);
            finalizeAction();
        }
    };

    window.deleteMessage = function(id) {
        inboxMessages = inboxMessages.filter(m => m.id !== id);
        spamMessages = spamMessages.filter(m => m.id !== id);
        finalizeAction();
    };

    window.clearCurrentList = function() {
        if (!confirm("Are you sure you want to delete all messages in this view?")) return;
        
        if (activeTab === 'inbox') {
            inboxMessages = [];
        } else if (activeTab === 'spam') {
            spamMessages = [];
        } else {
            inboxMessages = [];
            spamMessages = [];
        }
        
        finalizeAction();
    };

    function finalizeAction() {
        saveMessages();
        updateCounts();
        selectedId = null;
        switchTab(activeTab);
    }

    // 8. Boot Sequence
    loadMessages();
    updateCounts();

    if (serverPrediction && serverPrediction !== 'unknown') {
        saveMessages();
        updateCounts();
        switchTab(serverPrediction === 'spam' ? 'spam' : 'inbox');
    } else {
        switchTab('compose');
    }
});
```

---

## static/style.css

```css
/* ============================================
   MAIL GUARD - UI/UX STYLESHEET (DARK THEME)
   ============================================ */

:root {
    --bg-main: #0f172a;
    --bg-surface: #1e293b;
    --border-color: #334155;
    --text-primary: #f8fafc;
    --text-secondary: #cbd5e1;
    --text-muted: #64748b;
    
    --color-ham-bg: rgba(34, 197, 94, 0.1);
    --color-ham-border: rgba(34, 197, 94, 0.2);
    --color-ham-text: #4ade80;
    --color-ham-icon: rgba(34, 197, 94, 0.15);
    
    --color-spam-bg: rgba(239, 68, 68, 0.1);
    --color-spam-border: rgba(239, 68, 68, 0.2);
    --color-spam-text: #f87171;
    --color-spam-icon: rgba(239, 68, 68, 0.15);
}

* { 
    margin: 0; 
    padding: 0; 
    box-sizing: border-box; 
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg-main);
    color: var(--text-primary);
    min-height: 100vh;
}

/* ── Header ── */
.header {
    background: var(--bg-surface);
    border-bottom: 1px solid var(--border-color);
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    position: sticky;
    top: 0;
    z-index: 50;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.header-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}
.header-logo {
    width: 30px; 
    height: 30px;
    background: #3b82f6;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.header-logo svg { width: 16px; height: 16px; }
.header-title {
    font-size: 15px;
    font-weight: 700;
    letter-spacing: -0.3px;
}
.header-badge {
    font-size: 11px;
    background: var(--color-ham-bg);
    color: var(--color-ham-text);
    padding: 2px 8px;
    border-radius: 20px;
    font-weight: 600;
    border: 1px solid var(--color-ham-border);
}

/* ── Layout ── */
.app-layout {
    display: flex;
    height: calc(100vh - 56px);
    max-width: 1400px;
    margin: 0 auto;
}

/* ── Sidebar ── */
.sidebar {
    width: 220px;
    background: var(--bg-surface);
    border-right: 1px solid var(--border-color);
    padding: 16px 12px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
}
.sidebar-nav { display: flex; flex-direction: column; gap: 4px; }
.nav-btn {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 12px;
    border-radius: 10px;
    border: none;
    background: transparent;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
    transition: all 0.15s ease;
    width: 100%;
    text-align: left;
}
.nav-btn:hover { background: var(--bg-main); }
.nav-btn.active { background: #3b82f6; color: #fff; }
.nav-btn-left {
    display: flex;
    align-items: center;
    gap: 10px;
}
.nav-btn-left svg { width: 16px; height: 16px; flex-shrink: 0; }
.nav-count {
    font-size: 11px;
    font-weight: 700;
    padding: 1px 7px;
    border-radius: 20px;
    background: var(--bg-main);
    color: var(--text-secondary);
}
.nav-btn.active .nav-count { background: rgba(255,255,255,0.2); color: #fff; }
.nav-count.spam-count { background: var(--color-spam-bg); color: var(--color-spam-text); }
.nav-btn.active .nav-count.spam-count { background: rgba(255,255,255,0.2); color: #fff; }

.sidebar-divider {
    height: 1px;
    background: var(--border-color);
    margin: 16px 0;
}
.sidebar-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted);
    padding: 0 12px;
    margin-bottom: 8px;
}

/* ── Message List ── */
.message-panel {
    width: 380px;
    background: var(--bg-surface);
    border-right: 1px solid var(--border-color);
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
}
.message-panel-header {
    padding: 20px 20px 16px;
    border-bottom: 1px solid var(--bg-main);
}
.message-panel-title {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.3px;
}
.message-panel-sub {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 2px;
}
.message-list {
    flex: 1;
    overflow-y: auto;
}
.message-item {
    display: flex;
    gap: 12px;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border-color);
    cursor: pointer;
    transition: background 0.12s;
    position: relative;
}
.message-item:hover { background: var(--bg-main); }
.message-item.selected { background: rgba(255, 255, 255, 0.05); }

.msg-avatar {
    width: 36px; height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 700;
    color: #fff;
    flex-shrink: 0;
    margin-top: 2px;
}
.msg-content { flex: 1; min-width: 0; }
.msg-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
}
.msg-sender {
    font-size: 13px;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    color: var(--text-primary);
}
.msg-sender.read { font-weight: 500; color: var(--text-secondary); }
.msg-time {
    font-size: 11px;
    color: var(--text-muted);
    flex-shrink: 0;
}
.msg-subject {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-top: 2px;
}
.msg-preview {
    font-size: 12px;
    color: var(--text-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-top: 3px;
}
.msg-unread-dot {
    width: 8px; height: 8px;
    background: #3b82f6;
    border-radius: 50%;
    flex-shrink: 0;
    margin-top: 6px;
}

.empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
    height: 100%;
}
.empty-state svg { margin-bottom: 12px; opacity: 0.4; }
.empty-state p { font-size: 13px; font-weight: 500; }
.empty-state span { font-size: 12px; color: var(--border-color); margin-top: 4px; }

/* ── Detail / Compose Panel ── */
.detail-panel {
    flex: 1;
    background: var(--bg-surface);
    display: flex;
    flex-direction: column;
}

.compose-area {
    padding: 32px 40px;
    border-bottom: 1px solid var(--bg-main);
}
.compose-label {
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.compose-label svg { opacity: 0.4; }
.compose-textarea {
    width: 100%;
    min-height: 140px;
    padding: 16px;
    border: 1.5px solid var(--border-color);
    border-radius: 14px;
    font-size: 14px;
    font-family: inherit;
    color: var(--text-primary);
    resize: vertical;
    line-height: 1.65;
    background: var(--bg-main);
    transition: all 0.2s ease;
}
.compose-textarea:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.15);
    background: var(--bg-surface);
}
.compose-textarea::placeholder { color: var(--text-muted); }
.compose-actions {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 14px;
}
.btn-classify {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 22px;
    background: #3b82f6;
    color: #fff;
    border: none;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s, transform 0.1s;
}
.btn-classify:hover { background: #2563eb; }
.btn-classify:active { transform: scale(0.97); }
.btn-classify svg { width: 15px; height: 15px; }
.compose-hint {
    font-size: 12px;
    color: var(--text-muted);
}

/* Result Card */
.result-card {
    margin: 24px 40px;
    border-radius: 16px;
    overflow: hidden;
    animation: slideUp 0.35s ease;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
.result-card.ham { background: var(--color-ham-bg); border: 1.5px solid var(--color-ham-border); }
.result-card.spam { background: var(--color-spam-bg); border: 1.5px solid var(--color-spam-border); }

.result-header {
    padding: 20px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.result-label {
    display: flex;
    align-items: center;
    gap: 10px;
}
.result-icon {
    width: 38px; height: 38px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.result-icon.ham { background: var(--color-ham-icon); color: var(--color-ham-text); }
.result-icon.spam { background: var(--color-spam-icon); color: var(--color-spam-text); }
.result-icon svg { width: 18px; height: 18px; }
.result-text h3 { font-size: 15px; font-weight: 700; letter-spacing: -0.2px; }
.result-text h3.ham { color: var(--color-ham-text); }
.result-text h3.spam { color: var(--color-spam-text); }
.result-text p { font-size: 12px; margin-top: 1px; }
.result-text p.ham { color: var(--color-ham-text); opacity: 0.8; }
.result-text p.spam { color: var(--color-spam-text); opacity: 0.8; }
.result-confidence { text-align: right; }
.confidence-value { font-size: 28px; font-weight: 800; letter-spacing: -1px; }
.confidence-value.ham { color: var(--color-ham-text); }
.confidence-value.spam { color: var(--color-spam-text); }
.confidence-label { font-size: 11px; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }

.result-details { padding: 0 24px 20px; }
.confidence-bars { display: flex; gap: 12px; }
.conf-bar-group { flex: 1; }
.conf-bar-label { display: flex; justify-content: space-between; font-size: 11px; font-weight: 600; margin-bottom: 6px; }
.conf-bar-label .label-name { color: var(--text-secondary); }
.conf-bar-label .label-val { color: var(--text-primary); }
.conf-bar-track { height: 6px; background: rgba(255,255,255,0.05); border-radius: 10px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 10px; transition: width 0.6s ease; }
.conf-bar-fill.green { background: #4ade80; }
.conf-bar-fill.red { background: #f87171; }

/* Detail Email View */
.detail-email-header-wrap {
    padding: 20px 32px; 
    border-bottom: 1px solid var(--bg-main); 
    display:flex; 
    justify-content:space-between; 
    align-items:flex-start;
}
.detail-email {
    flex: 1;
    overflow-y: auto;
    padding: 32px 40px;
}
.detail-email-subject {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.3px;
    margin-bottom: 12px;
}
.detail-email-meta { display: flex; align-items: center; gap: 10px; }
.meta-sender { font-size: 13px; font-weight: 500; color: var(--text-primary); }
.meta-time { font-size: 12px; color: var(--text-muted); }
.meta-badge { font-size: 11px; padding: 2px 8px; border-radius: 20px; font-weight: 600; margin-left: 4px; border: 1px solid transparent; }
.detail-email-body {
    font-size: 14px;
    line-height: 1.75;
    color: var(--text-secondary);
    white-space: pre-line;
    max-width: 580px;
}
.spam-warning {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 14px 18px;
    background: var(--color-spam-bg);
    border: 1px solid var(--color-spam-border);
    border-radius: 12px;
    margin-bottom: 20px;
}
.spam-warning svg { width: 16px; height: 16px; color: var(--color-spam-text); flex-shrink: 0; }
.spam-warning p { font-size: 12px; color: var(--color-spam-text); font-weight: 500; }

.detail-actions { display: flex; gap: 8px; margin-top: 8px; }
.btn-action {
    padding: 6px 14px;
    border-radius: 8px;
    border: none;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    font-family: inherit;
    transition: all 0.12s;
}
.btn-action.move-spam { background: var(--color-spam-bg); color: var(--color-spam-text); border: 1px solid var(--color-spam-border); }
.btn-action.move-spam:hover { background: rgba(239, 68, 68, 0.2); }
.btn-action.move-inbox { background: var(--color-ham-bg); color: var(--color-ham-text); border: 1px solid var(--color-ham-border); }
.btn-action.move-inbox:hover { background: rgba(34, 197, 94, 0.2); }
.btn-action.delete { background: var(--bg-main); color: var(--text-secondary); border: 1px solid var(--border-color); }
.btn-action.delete:hover { background: var(--border-color); color: var(--text-primary); }

.detail-placeholder {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
}
.detail-placeholder svg { margin-bottom: 16px; opacity: 0.35; }
.detail-placeholder p { font-size: 14px; font-weight: 500; color: var(--text-muted); }
.detail-placeholder span { font-size: 12px; color: var(--border-color); margin-top: 4px; }

/* Scrollbars */
.message-list::-webkit-scrollbar,
.detail-email::-webkit-scrollbar { width: 6px; }
.message-list::-webkit-scrollbar-thumb,
.detail-email::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 10px; }

/* ── Responsive ── */
@media (max-width: 1024px) {
    .sidebar { width: 64px; padding: 16px 8px; }
    .nav-btn span.nav-label, .nav-count, .sidebar-label { display: none; }
    .nav-btn { justify-content: center; padding: 10px; }
    .nav-btn-left { gap: 0; }
    .sidebar-divider { margin: 12px 4px; }
}
@media (max-width: 768px) {
    .message-panel { width: 100%; }
    .detail-panel { display: none; }
    .detail-panel.mobile-show {
        display: flex;
        position: fixed;
        inset: 56px 0 0 0;
        z-index: 40;
    }
}
```

---

## templates/index.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mail Guard</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>

<div id="app-data"
     data-prediction="{{ prediction if prediction else '' }}"
     data-email-text="{{ email_text if email_text else '' }}"
     data-confidence="{{ confidence if confidence else 0 }}"
     data-ham-confidence="{{ ham_confidence if ham_confidence else 0 }}"
     data-spam-confidence="{{ spam_confidence if spam_confidence else 0 }}"
     style="display: none;"></div>

<header class="header">
    <div class="header-brand">
        <div class="header-logo">
            <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <rect x="2" y="4" width="20" height="16" rx="2"/>
                <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>
            </svg>
        </div>
        <span class="header-title">Mail Guard</span>
        <span class="header-badge">AI Powered</span>
    </div>
</header>

<div class="app-layout">

    <aside class="sidebar">
        <nav class="sidebar-nav">
            <button class="nav-btn active" id="nav-inbox" onclick="switchTab('inbox')">
                <span class="nav-btn-left">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="2" y="4" width="20" height="16" rx="2"/>
                        <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>
                    </svg>
                    <span class="nav-label">Inbox</span>
                </span>
                <span class="nav-count" id="inbox-count">0</span>
            </button>
            <button class="nav-btn" id="nav-spam" onclick="switchTab('spam')">
                <span class="nav-btn-left">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/>
                    </svg>
                    <span class="nav-label">Spam</span>
                </span>
                <span class="nav-count spam-count" id="spam-count">0</span>
            </button>
        </nav>

        <div class="sidebar-divider"></div>
        <p class="sidebar-label">Classify</p>
        <button class="nav-btn" id="nav-compose" onclick="switchTab('compose')">
            <span class="nav-btn-left">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/>
                </svg>
                <span class="nav-label">New Check</span>
            </span>
        </button>
    </aside>

    <div class="message-panel">
        <div class="message-panel-header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h2 class="message-panel-title" id="panel-title">Inbox</h2>
                <button id="clear-list-btn" onclick="clearCurrentList()" style="display:none; background:transparent; border:none; color:#ef4444; font-size:12px; font-weight:600; cursor:pointer;">Clear All</button>
            </div>
            <p class="message-panel-sub" id="panel-sub">0 messages</p>
        </div>
        <div class="message-list" id="message-list">
            <div class="empty-state" id="empty-state">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="2" y="4" width="20" height="16" rx="2"/>
                    <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>
                </svg>
                <p>No messages yet</p>
                <span>Classify an email to get started</span>
            </div>
        </div>
    </div>

    <div class="detail-panel" id="detail-panel">

        <div id="compose-view">
            <form method="POST" action="/">
                <div class="compose-area">
                    <label class="compose-label">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/>
                        </svg>
                        Paste email content to classify
                    </label>
                    <textarea
                        name="email_text"
                        class="compose-textarea"
                        placeholder="Paste the email text here and click classify to check if it's spam or not..."
                        rows="6"
                        required
                    >{{ email_text if email_text else '' }}</textarea>
                    <div class="compose-actions">
                        <button type="submit" class="btn-classify">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="m22 2-7 20-4-9-9-4z"/><path d="M22 2 11 13"/>
                            </svg>
                            Classify Email
                        </button>
                        <span class="compose-hint">Powered by ML model</span>
                    </div>
                </div>
            </form>

            {% if prediction and prediction != 'unknown' %}
            <div class="result-card {{ prediction }}">
                <div class="result-header">
                    <div class="result-label">
                        <div class="result-icon {{ prediction }}">
                            {% if prediction == 'ham' %}
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M20 6 9 17l-5-5"/>
                            </svg>
                            {% elif prediction == 'spam' %}
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                                <circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/>
                            </svg>
                            {% endif %}
                        </div>
                        <div class="result-text">
                            <h3 class="{{ prediction }}">
                                {% if prediction == 'ham' %}Not Spam — Safe
                                {% elif prediction == 'spam' %}Spam Detected
                                {% endif %}
                            </h3>
                            <p class="{{ prediction }}">
                                {% if prediction == 'ham' %}This message looks legitimate
                                {% elif prediction == 'spam' %}This message appears to be spam
                                {% endif %}
                            </p>
                        </div>
                    </div>
                    <div class="result-confidence">
                        <div class="confidence-value {{ prediction }}">{{ confidence }}%</div>
                        <div class="confidence-label">Confidence</div>
                    </div>
                </div>

                <div class="result-details">
                    <div class="confidence-bars">
                        <div class="conf-bar-group">
                            <div class="conf-bar-label">
                                <span class="label-name">Safe</span>
                                <span class="label-val">{{ ham_confidence }}%</span>
                            </div>
                            <div class="conf-bar-track">
                                <div class="conf-bar-fill green" style="width: {{ ham_confidence }}%"></div>
                            </div>
                        </div>
                        <div class="conf-bar-group">
                            <div class="conf-bar-label">
                                <span class="label-name">Spam</span>
                                <span class="label-val">{{ spam_confidence }}%</span>
                            </div>
                            <div class="conf-bar-track">
                                <div class="conf-bar-fill red" style="width: {{ spam_confidence }}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            {% endif %}

            {% if not prediction or prediction == 'unknown' %}
            <div class="detail-placeholder">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">
                    <path d="m22 2-7 20-4-9-9-4z"/><path d="M22 2 11 13"/>
                </svg>
                <p>Paste an email above and hit classify</p>
                <span>The AI will sort it into Inbox or Spam for you</span>
            </div>
            {% endif %}
        </div>

        <div id="email-detail-view" style="display:none; flex-direction:column; flex:1;">
            <div class="detail-email-header-wrap">
                <div>
                    <div class="detail-email-subject" id="detail-subject"></div>
                    <div class="detail-email-meta">
                        <div class="msg-avatar" id="detail-avatar" style="width:28px;height:28px;font-size:10px;"></div>
                        <span class="meta-sender" id="detail-sender"></span>
                        <span class="meta-time" id="detail-time"></span>
                        <span class="meta-badge" id="detail-badge"></span>
                    </div>
                </div>
                <div class="detail-actions" id="detail-actions"></div>
            </div>
            <div class="detail-email" id="detail-body-container">
                <div id="detail-warning"></div>
                <div class="detail-email-body" id="detail-body"></div>
            </div>
        </div>
    </div>
</div>

<script src="{{ url_for('static', filename='main.js') }}"></script>
</body>
</html>
```

---

## requirements.txt

```plaintext
flask==3.1.1
scikit-learn==1.8.0
pandas==2.2.3
nltk==3.9.1
joblib==1.4.2
```

---

## pyproject.toml

```plaintext
[project]
name = "spam-classifier"
version = "0.1.0"
requires-python = ">=3.14"
dependencies = []
```

---

## MailGuard.spec

```plaintext
# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[('templates', 'templates'), ('static', 'static'), ('models', 'models'), ('/home/codespace/nltk_data', 'nltk_data')],
    hiddenimports=['sklearn.naive_bayes', 'sklearn.feature_extraction.text', 'nltk', 'joblib'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MailGuard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
```

---

## Summary

This dump file contains all the source code from the EmailSpamDetector repository:
- **Backend**: Flask web application with ML-powered spam detection
- **Frontend**: React-like vanilla JavaScript with dark theme UI
- **Models**: Scikit-learn Naive Bayes classifier with TF-IDF vectorizer
- **Data Processing**: Text preprocessing with NLTK (stemming, tokenization, stopword removal)
- **Build System**: PyInstaller configuration for standalone executable

**Total Lines of Code**: ~1,500+ lines across Python, JavaScript, CSS, and HTML files.
