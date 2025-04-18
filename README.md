# 🎤 Speech-to-Mood Recognizer

A unique Python project that detects a user's **mood** based on their **voice tone and content**. It uses real-time **speech recognition**, extracts **textual sentiment**, and classifies the **emotional state** using a trained ML model.

## 🚀 Features
- 🎙️ Converts real-time speech to text using Google Speech API
- 😌 Classifies the mood into Happy, Sad, Angry, or Neutral
- 🧠 Uses an SVM classifier trained on labeled emotional data
- 📊 Clean CLI interface with instant results

## 📦 Requirements
- Python 3.7+
- `speechrecognition`
- `scikit-learn`
- `pyaudio`
- `nltk`

Install them via:
```bash
pip install speechrecognition scikit-learn pyaudio nltk
