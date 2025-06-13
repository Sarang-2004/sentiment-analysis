# Sentiment Analysis of Movie Reviews 🎬🧠

This project implements a sentiment analysis model from scratch using Python. It classifies movie reviews as **positive** or **negative** based on textual content, using the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

## 🔍 Project Objective

To build a machine learning model that can understand the sentiment behind movie reviews using NLP (Natural Language Processing) techniques and evaluate its performance.

---

## 📁 Dataset

- Source: [Kaggle - IMDB Movie Review Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Size: 50,000 reviews (25k for training, 25k for testing)
- Labels: `positive` and `negative`
- Balanced dataset (50% positive, 50% negative)

---

## ⚙️ Tech Stack

- **Language:** Python
- **Libraries:** 
  - `pandas` and `numpy` for data manipulation
  - `scikit-learn` for preprocessing and model building
  - `nltk` for NLP tasks (tokenization, stopwords removal)
  - `matplotlib` and `seaborn` for visualization

---

## 🚀 Features

- Preprocesses raw text (lowercasing, punctuation removal, stopword filtering, etc.)
- Converts text to numerical features using:
  - TF-IDF 
- Trains models like:
  - Logistic Regression
- Evaluation using:
  - Accuracy
  - Confusion Matrix
  - Precision, Recall, F1 Score

---

## 🧪 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Sarang-2004/sentiment-analysis.git
cd sentiment-analysis 
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the Code
```bash
python sentiment_analysis.py
```
---

##📊 Sample Output
✅ Accuracy: ~85% (depends on model and preprocessing)

🔍 Confusion Matrix to analyze classification

📈 Graphs showing model performance and feature distributions

---

##🧠 Learnings
-Basics of Natural Language Processing (NLP)

-Text preprocessing techniques

-Building and evaluating ML models on text data

-Importance of feature extraction like BoW and TF-IDF

---

##📌 To-Do
 -Implement deep learning model (e.g., LSTM or BERT)

 -Add more preprocessing techniques like stemming/lemmatization

---

##📜 License
-This project is licensed under the MIT License.


