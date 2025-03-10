from sklearn.feature_extraction.text import CountVectorizer
import joblib

def load_model_and_vectorizer(model_path='sentiment_model.pkl', vectorizer_path='vectorizer.pkl'):
    """Load a trained model and vectorizer from disk."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_sentiment(model, vectorizer, text):
    """Predict the sentiment of a single text input."""
    text_vector = vectorizer.transform([text])  # Transform text to vector
    prediction = model.predict(text_vector)  # Predict sentiment
    return 'Positive' if prediction[0] == 1 else 'Negative'  # Return sentiment based on prediction
