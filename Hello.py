import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 1. Dataset path
data_path = 'Audio_Speech_Actors_01-24'

# 2. Feature Extraction Function
def extract_features(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

    return np.hstack([
        np.mean(mfccs.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(mel.T, axis=0),
        np.mean(contrast.T, axis=0),
        np.mean(tonnetz.T, axis=0),
    ])

# 3. RAVDESS Emotion Mapping
emotion_dict = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# 4. Loop over all Actor folders and extract features
features_list = []
labels = []

for actor_folder in os.listdir(data_path):
    actor_path = os.path.join(data_path, actor_folder)
    if not os.path.isdir(actor_path):
        continue
    for file in os.listdir(actor_path):
        if file.endswith(".wav"):
            file_path = os.path.join(actor_path, file)
            try:
                emotion_code = file.split('-')[2]
                emotion = emotion_dict.get(emotion_code)
                if emotion is None:
                    continue
                features = extract_features(file_path)
                features_list.append(features)
                labels.append(emotion)
            except Exception as e:
                print(f"Error with file {file_path}: {e}")

# 5. Prepare Data
X = np.array(features_list)
y = LabelEncoder().fit_transform(labels)
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Build Model
model = Sequential([
    Dense(256, input_shape=(X.shape[1],), activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
model.summary()

# 7. Train
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, batch_size=32)

# 8. Plot
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Model Accuracy over Epochs")
plt.grid(True)
plt.show()

# 9. Save Model
model.save("speech_emotion_recognition_model.h5")