import sys
import joblib
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import json  # For JSON formatting

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)

# Load the saved model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_sentiment(text):
    """Predict the sentiment of a single text input."""
    # Preprocess the text
    text_vector = vectorizer.transform([text]).toarray()
    prediction = model.predict(text_vector)[0]
    
    # Return the sentiment as a string
    return "Positive" if prediction == 1 else "Negative"

if __name__ == "__main__":
    # Handle input text
    if len(sys.argv) > 1:
        input_text = sys.argv[1]
    else:
        input_text = input("Enter the text to analyze: ")

    # Get sentiment prediction
    sentiment_result = predict_sentiment(input_text)
    
    # Format the result as JSON
    result_json = {
        "Sentiment": sentiment_result
    }
    
    # Print the JSON result
    print(json.dumps(result_json))
