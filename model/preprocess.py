import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import nltk

nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize


def clean_text(text):
    """
    Remove special characters, handle negations, and lowercase the text.
    """
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase
    tokens = [word for word in tokens if word.isalnum()]  # Remove non-alphanumeric tokens
    return ' '.join(tokens)


def load_and_preprocess_data(filepath):
    """
    Load the dataset and preprocess it.
    """
    # Load dataset with specific encoding
    data = pd.read_csv(filepath, encoding='latin1')
    # Check for missing values
    data.dropna(subset=['review', 'sentiment'], inplace=True)
    # Clean text data
    data['review'] = data['review'].apply(clean_text)
    # Encode sentiment as binary
    X = data['review']
    y = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    # Split dataset
    return train_test_split(X, y, test_size=0.2, random_state=42)


def vectorize_text(X_train, X_test, save_vectorizer=True):
    """
    Convert text data into numerical format using TfidfVectorizer.
    """
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Use n-grams for better context
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()

    if save_vectorizer:
        # Save the fitted vectorizer
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)

    return X_train_vec, X_test_vec, vectorizer


def load_vectorizer(filepath='vectorizer.pkl'):
    """
    Load a saved TfidfVectorizer.
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# Main execution
if __name__ == "__main__":
    # Filepath to your dataset
    filepath = r'D:\Sarang\sentiment analysis\IMDB Dataset.csv'
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)
    
    # Vectorize text
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)

    # Display some information
    print("Shape of training data:", X_train_vec.shape)
    print("Shape of test data:", X_test_vec.shape)
