import pickle

# Load the model from the .pkl file
with open('sentiment_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Check the type of the model to understand what it is
print(type(model))
