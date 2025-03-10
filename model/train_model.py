from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # For saving and loading the model


def train_and_save_model(X_train, y_train, model_path='sentiment_model.pkl'):
    """
    Train a Logistic Regression model and save it to a file.

    Parameters:
        X_train: Training feature set
        y_train: Training labels
        model_path: Path to save the trained model

    Returns:
        model: The trained Logistic Regression model
    """
    # Create and train the Logistic Regression model
    model = LogisticRegression(max_iter=1000)  # Adjust max_iter for convergence if needed
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    return model


def load_model(model_path='sentiment_model.pkl'):
    """
    Load a saved Logistic Regression model.

    Parameters:
        model_path: Path to the saved model

    Returns:
        model: The loaded Logistic Regression model
    """
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.

    Parameters:
        model: Trained model to evaluate
        X_test: Test feature set
        y_test: Test labels

    Returns:
        accuracy: The accuracy of the model
    """
    predictions = model.predict(X_test)

    # Print evaluation metrics
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:\n", classification_report(y_test, predictions))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))

    return accuracy_score(y_test, predictions)


# Main execution for demonstration
if __name__ == "__main__":
    # Ensure X_train, X_test, y_train, y_test are loaded/preprocessed beforehand
    from preprocess import load_and_preprocess_data, vectorize_text

    # Load and preprocess data
    filepath = r'D:\Sarang\sentiment analysis\IMDB Dataset.csv'
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)
    X_train_vec, X_test_vec, _ = vectorize_text(X_train, X_test)

    # Train and save the model
    model = train_and_save_model(X_train_vec, y_train)

    # Load the model and evaluate
    loaded_model = load_model()
    evaluate_model(loaded_model, X_test_vec, y_test)
