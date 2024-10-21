import os
import numpy as np
import librosa
import pandas as pd

true_files = '/Users/30068385/OneDrive - Western Sydney University/Audio Dataset/1min/True'
false_files = '/Users/30068385/OneDrive - Western Sydney University/Audio Dataset/1min/False'


def extract_features(audio, sample_rate):
    hop_length = sample_rate
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, hop_length=hop_length, n_mels=40, fmin=250, fmax=2000)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    # Normalizar entre 0 y 1
    normalized_spectrogram = (log_spectrogram - log_spectrogram.min()) / (log_spectrogram.max() - log_spectrogram.min())
    return normalized_spectrogram.flatten()  # Aplanar para guardar como CSV

features_list = []
labels_list = []

for file in os.listdir(true_files):
    if file.endswith('.wav'):
        file_path = os.path.join(true_files, file)
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        frame_length = int(0.004 * sample_rate)  # 4 ms en muestras
        hop_length = frame_length
        for i in range(0, len(audio) - frame_length, hop_length):
            segment = audio[i:i + frame_length]
            features = extract_features(segment, sample_rate)
            features_list.append(features)
            labels_list.append(1)

# Procesar archivos "False"
for file in os.listdir(false_files):
    if file.endswith('.wav'):
        file_path = os.path.join(false_files, file)
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        frame_length = int(0.004 * sample_rate)  # 4 ms en muestras
        hop_length = frame_length  # Sin solapamiento, cada segmento es de 4 ms
        for i in range(0, len(audio) - frame_length, hop_length):
            segment = audio[i:i + frame_length]
            features = extract_features(segment, sample_rate)
            features_list.append(features)
            labels_list.append(0)

df = pd.DataFrame(features_list)
df['label'] = labels_list
df.to_csv('/Users/30068385/OneDrive - Western Sydney University/Audio Dataset/1min/feature_data.csv', index=False)