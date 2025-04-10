import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def extract_features(file):
    y, sr = librosa.load(file, duration=5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    return np.hstack([mfccs, chroma, contrast])

# Label mapping
labels_map = {
    "happy": 0,
    "neutral": 1,
    "sad": 2
}

X, y = [], []

# Load training samples
for file in os.listdir("audio_samples"):
    for mood in labels_map:
        if mood in file:
            features = extract_features(f"audio_samples/{file}")
            X.append(features)
            y.append(labels_map[mood])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/mood_classifier.pkl")

# Evaluate
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")
