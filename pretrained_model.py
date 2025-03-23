import numpy as np
import librosa
import pickle
from tensorflow.keras.models import model_from_json
import librosa.display
from collections import Counter

# Load the saved model structure
with open('/content/drive/MyDrive/all files for SER of linux/CNN_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Load the model
loaded_model = model_from_json(loaded_model_json)

# Load the model weights
loaded_model.load_weights('/content/drive/MyDrive/all files for SER of linux/CNN_model_weights.weights.h5')
print("Model loaded successfully.")

# Load the scaler
with open('/content/drive/MyDrive/all files for SER of linux/scaler2.pickle', 'rb') as f:
    scaler = pickle.load(f)

# Load the encoder
with open('/content/drive/MyDrive/all files for SER of linux/encoder2.pickle', 'rb') as f:
    encoder = pickle.load(f)

# Compile the model
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Feature extraction functions
def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)
                        ))
    return result

# Function to preprocess audio files with dynamic padding
def get_predict_feat(data, expected_length=2376):
    features = extract_features(data)
    # Adjust feature length
    if len(features) < expected_length:
        features = np.pad(features, (0, expected_length - len(features)))  # Pad with zeros
    else:
        features = features[:expected_length]  # Truncate if too long
    
    features = np.array(features)
    features = np.reshape(features, newshape=(1, expected_length))
    scaled_features = scaler.transform(features)
    final_features = np.expand_dims(scaled_features, axis=2)
    return final_features

# Prediction function
def predict_emotion(audio_path):
    # Load the audio file
    data, sr = librosa.load(audio_path, sr=22050)
    segment_length = 5  # seconds
    hop_length = sr * segment_length

    # Process the audio file in segments
    emotions = []
    for i in range(0, len(data), hop_length):
        segment = data[i:i+hop_length]
        if len(segment) < hop_length:
            break  # Ignore segments shorter than required length
        features = get_predict_feat(segment)
        prediction = loaded_model.predict(features)
        predicted_emotion = encoder.inverse_transform(prediction)[0][0]
        emotions.append(predicted_emotion)

    # Print all recorded emotions
    print(f"Recorded Emotions: {emotions}")

    # Determine the majority emotion
    final_emotion = Counter(emotions).most_common(1)[0][0]
    print(f"Predicted Emotion: {final_emotion}")

# Test the model with a sample audio file
audio_file_path = '/content/Beijing faces heat over Uighurs abuse, shocking images surface ｜ Latest English News ｜ WION.wav'
predict_emotion(audio_file_path)
