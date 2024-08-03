import os
import zipfile
import librosa
import numpy as np
import pickle
import ast

def extract_zip_file(file_path, data_path):
    unzip_path = data_path
    os.makedirs(unzip_path, exist_ok=True)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_path)


def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=50)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    return mfccs_scaled_features
    

def load_encoder(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_encoder(encoder, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(encoder, f)


def convert_to_array(str_array):
    return np.array(ast.literal_eval(str_array), dtype=np.float32)




