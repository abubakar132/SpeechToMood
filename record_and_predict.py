import speech_recognition as sr
import librosa
import numpy as np
import joblib
import soundfile as sf

def extract_features_from_audio(file):
    y, sr = librosa.load(file, duration=5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    return np.hstack([mfccs, chroma, contrast])

def record_audio(output="live_input.wav"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Speak now (5 seconds)...")
        audio = r.record(source, duration=5)
        with open(output, "wb") as f:
            f.write(audio.get_wav_data())
        print("✅ Recording done!")

record_audio()

features = extract_features_from_audio("live_input.wav")
model = joblib.load("model/mood_classifier.pkl")
prediction = model.predict([features])[0]
mood_map = {0: "😊 Happy", 1: "😐 Neutral", 2: "😢 Sad"}
print(f"🧠 Detected Mood: {mood_map[prediction]}")
