import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the model and vectorizer
model = joblib.load('emotion_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Test sentences with different emotions
test_sentences = [
    "I feel so happy and excited today!",
    "I am really sad and disappointed",
    "I'm angry about this situation",
    "I love this so much",
    "I'm frightened and scared",
    "This is surprising and amazing",
    "I feel terrible and hopeless",
    "I'm absolutely delighted",
    "This makes me furious",
    "I feel safe and loved"
]

print("=" * 80)
print("EMOTION PREDICTION TEST")
print("=" * 80)

for sentence in test_sentences:
    vec = vectorizer.transform([sentence])
    pred = model.predict(vec)
    proba = model.predict_proba(vec)[0]
    
    print(f"\nText: {sentence}")
    print(f"Predicted Emotion: {pred[0].upper()}")
    print("Confidence Scores:")
    for emotion, score in zip(model.classes_, proba):
        print(f"  {emotion:10s}: {score:.4f}")

# Evaluate on test set
print("\n" + "=" * 80)
print("FULL MODEL EVALUATION")
print("=" * 80)

# Load full dataset to evaluate
data = []
with open('train.txt', 'r') as f:
    for line in f:
        if ';' in line:
            text, label = line.rsplit(';', 1)
            data.append({'text': text.strip(), 'label': label.strip()})

df = pd.DataFrame(data)

# Get all predictions
X_all = df['text']
y_all = df['label']

X_all_vec = vectorizer.transform(X_all)
y_pred_all = model.predict(X_all_vec)

print("\nOverall Classification Report:")
print(classification_report(y_all, y_pred_all))

# Confusion Matrix
cm = confusion_matrix(y_all, y_pred_all)
print("\nConfusion Matrix:")
print(cm)

# Save confusion matrix plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Emotion Prediction Confusion Matrix')
plt.ylabel('True Emotion')
plt.xlabel('Predicted Emotion')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix plot saved as 'confusion_matrix.png'")

# Accuracy per emotion
print("\nAccuracy per Emotion:")
for emotion in model.classes_:
    mask = y_all == emotion
    accuracy = (y_pred_all[mask] == y_all[mask]).sum() / mask.sum()
    print(f"  {emotion:10s}: {accuracy:.2%}")
