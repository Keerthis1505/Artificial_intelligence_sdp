import joblib

# Load the model and vectorizer
model = joblib.load('emotion_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Example prediction
text = "I feel so happy today!"
vec = vectorizer.transform([text])
pred = model.predict(vec)
print(f"Predicted emotion: {pred[0]}")