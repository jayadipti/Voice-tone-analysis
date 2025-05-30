import numpy as np
from tensorflow.keras.models import load_model
import librosa

model = load_model("speech_emotion_recognition_model.h5")

emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

    features = np.hstack([
        np.mean(mfccs.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(mel.T, axis=0),
        np.mean(contrast.T, axis=0),
        np.mean(tonnetz.T, axis=0),
    ])
    return features


file_path = r"Audio_Speech_Actors_01-24/Actor_07/03-01-03-01-02-02-07.wav"


features = extract_features(file_path).reshape(1, -1)


prediction = model.predict(features)
predicted_index = np.argmax(prediction)
predicted_emotion = emotion_labels[predicted_index]
confidence = prediction[0][predicted_index] * 100  


print(f"Predicted Emotion: {predicted_emotion}")
print(f"Confidence: {confidence:.2f}%")
