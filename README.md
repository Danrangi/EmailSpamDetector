# 📧 Mail Guard — AI Email Spam Detector

A clean, modern web application that classifies emails as **spam** or **safe** using machine learning. Built with Flask, scikit-learn, and a sleek three-panel inbox UI.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=flat&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## ✨ Features

- **AI-Powered Classification** — Uses a trained ML model with TF-IDF vectorization to detect spam
- **Three-Panel Inbox UI** — Sidebar navigation, message list, and detail view
- **Inbox & Spam Folders** — Classified emails are automatically sorted into the correct folder
- **Confidence Scores** — Visual confidence bars showing spam vs. safe probability
- **Message Management** — Move messages between folders, delete, and mark as read
- **Persistent Storage** — Messages are saved in the browser's localStorage across sessions
- **Responsive Design** — Works on desktop and adapts to smaller screens
- **Real-Time Feedback** — Instant classification results with animated result cards

---

## 📸 How It Works

1. **Paste** an email's text content into the compose area
2. **Click** "Classify Email" to run it through the ML model
3. **View** the result — the email is sorted into Inbox (safe) or Spam
4. **Manage** your messages — move between folders, read, or delete

---

## 🛠️ Tech Stack

| Layer        | Technology                          |
|-------------|--------------------------------------|
| **Backend**  | Python, Flask                       |
| **ML Model** | scikit-learn (TF-IDF + Classifier)  |
| **NLP**      | NLTK (tokenization, stemming, stopwords) |
| **Frontend** | HTML, CSS, Vanilla JavaScript       |
| **Storage**  | Browser localStorage (client-side)  |

---

## 📁 Project Structure
