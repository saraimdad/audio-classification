import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import convert_to_array, save_encoder

@dataclass
class ModelTrainingConfig:
    trained_model_path: str = os.path.join('artifacts', 'models', 'trained.keras')
    encoder_path: str = os.path.join('artifacts', 'encoders', 'label_encoder.pkl')
    EPOCHS: int = 100
    BATCH_SIZE: int = 32

class ModelTraining:
    def __init__(self):
        self.training_config = ModelTrainingConfig()

    def initiate_training(self, transformed_data_path, model_path):
        try: 
            logging.info('Training Initiated')
            audio_df = pd.read_csv(transformed_data_path)
            audio_df['feature'] = audio_df['feature'].apply(convert_to_array)

            X = np.array(audio_df['feature'].tolist())
            y = np.array(audio_df['class'].tolist())

            label_encoder = LabelEncoder()
            y = to_categorical(label_encoder.fit_transform(y))

            save_encoder(label_encoder, self.training_config.encoder_path)


            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # print(type(X), X.dtype)
            # print(type(y), y.dtype)
            # print(X_train.shape)
            # print(X_test.shape)
            # print(y_train.shape)
            # print(y_test.shape)

            checkpoint = ModelCheckpoint(filepath=self.training_config.trained_model_path, verbose=1, save_best_only=True)

            model = load_model(model_path)
            start = datetime.now()
            model.fit(X_train, y_train, batch_size=self.training_config.BATCH_SIZE, epochs=self.training_config.EPOCHS, validation_data=(X_test, y_test), callbacks=[checkpoint])
            duration = datetime.now() - start

            logging.info('Training Completed')
            print('Training completed in time: ', duration)

            return self.training_config.trained_model_path, label_encoder

        except Exception as e:
            raise CustomException(e, sys)