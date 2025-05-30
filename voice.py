# import numpy as np
# import librosa
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import LabelEncoder


# model = load_model("speech_emotion_recognition_model.h5")


# emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# le = LabelEncoder()
# le.fit(emotion_labels)


# def extract_features(file_path, sr=22050):
#     y, sr = librosa.load(file_path, sr=sr)
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
#     chroma = librosa.feature.chroma_stft(y=y, sr=sr)
#     mel = librosa.feature.melspectrogram(y=y, sr=sr)
#     contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
#     tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    
#     features = np.hstack([
#         np.mean(mfccs.T, axis=0),
#         np.mean(chroma.T, axis=0),
#         np.mean(mel.T, axis=0),
#         np.mean(contrast.T, axis=0),
#         np.mean(tonnetz.T, axis=0),
#     ])
#     return features


# file_path = "scream-with-echo-46585.mp3"


# features = extract_features(file_path)
# features_extracted = features.reshape(1, -1)

# # Ensure the model is compiled before prediction
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# prediction = model.predict(features_extracted)
# predicted_index = np.argmax(prediction)
# predicted_emotion = le.inverse_transform([predicted_index])[0]
# confidence = prediction[0][predicted_index] * 100  # Convert to percentage


# print(f"Predicted Emotion: {predicted_emotion}")
# print(f"Confidence: {confidence:.2f}%")


import os
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model
import tensorflow as tf


tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

AUDIO_FOLDER = 'Audio_Speech_Actors_01-24'  
MODEL_PATH = 'speech_emotion_recognition_model.h5'


emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
le = LabelEncoder()
le.fit(emotion_labels)


model = load_model(MODEL_PATH)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


def extract_features(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)
    
    
    print(f"\nProcessing: {file_path}")
    print("Audio Duration:", librosa.get_duration(y=y, sr=sr), "sec")
    plt.figure(figsize=(6, 2))
    plt.plot(y)
    plt.title(os.path.basename(file_path))
    plt.show()

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


    print("Feature mean:", np.mean(features))
    print("Feature std dev:", np.std(features))
    print("Feature shape:", features.shape)

    return features


results = []

for root, dirs, files in os.walk(AUDIO_FOLDER):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            try:
                features = extract_features(file_path)
                features_scaled = StandardScaler().fit_transform(features.reshape(1, -1))

                prediction = model.predict(features_scaled)
                predicted_index = np.argmax(prediction)
                predicted_emotion = le.inverse_transform([predicted_index])[0]
                confidence = round(float(prediction[0][predicted_index]) * 100, 2)

                print(f"Predicted Emotion: {predicted_emotion}, Confidence: {confidence}%")
                
                results.append({
                    'file': file,
                    'emotion': predicted_emotion,
                    'confidence (%)': confidence
                })

            except Exception as e:
                print(f"Error with file {file_path}: {e}")


df = pd.DataFrame(results)
df.to_csv("emotion_predictions.csv", index=False)
print("\nSaved predictions to emotion_predictions.csv")
