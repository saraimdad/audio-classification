import os
import sys
from dataclasses import dataclass
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

from src.exception import CustomException
from src.logger import logging


@dataclass
class PrepareModelConfig:
    model_path: str = os.path.join('artifacts', 'models', 'model.keras')


class PrepareModel():
    def __init__(self):
        self.preparation_config = PrepareModelConfig()

    def initiate_model_preparation(self):
        logging.info('Model Preparation Initiated')
        try:            
            model=Sequential()

            model.add(Dense(100,input_shape=(50,)))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))

            model.add(Dense(200))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))

            model.add(Dense(200))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))

            model.add(Dense(200))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))

            model.add(Dense(100))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))


            model.add(Dense(10))
            model.add(Activation('softmax'))

            model.summary()

            model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

            os.makedirs(os.path.dirname(self.preparation_config.model_path), exist_ok=True)
            model.save(self.preparation_config.model_path)         
            logging.info('Model Preparation Completed')

            return self.preparation_config.model_path

        except Exception as e:
            raise CustomException(e, sys)