import sys
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

from src.exception import CustomException
from src.utils import extract_features, load_encoder
from src.pipeline.train_pipeline import TrainPipeline

class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, audio_path):
        try:
            model_path = 'artifacts/models/trained.keras'
            encoder_path = 'artifacts/encoders/label_encoder.pkl'

            model = load_model(model_path)
            encoder = load_encoder(encoder_path)

            mfcc = extract_features(audio_path)

            pred = model.predict(np.array([mfcc]))
            pred_class = np.argmax(pred)

            label = encoder.inverse_transform([pred_class])
            label = label[0].replace('_', ' ').title()

            return label
        
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__=='__main__':
    train = TrainPipeline()
    train.train()

    pred_obj = PredictPipeline()
    prediction = pred_obj.predict('C:/Users/SaraImdad/Downloads/mixkit-happy-puppy-barks-741.wav')
    print(prediction)