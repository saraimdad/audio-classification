import os
import sys
import pandas as pd
import numpy as np
import librosa
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import tqdm
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import extract_features


@dataclass
class DataTransformationConfig:
    transformed_data_path: str = os.path.join('artifacts', 'transformed_data', 'transformed_data.csv')

class DataTransformation():
    def __init__(self):
        self.transformation_config = DataTransformationConfig()


    def initiate_data_transformation(self, data_path):
        logging.info('Data Transformation Initiated')
        try:
            extracted_features = []
            metadata = pd.read_csv(os.path.join(data_path, 'UrbanSound8k.csv'))

            for index_num, row in tqdm.tqdm(metadata.iterrows()):
                final_path = os.path.join(data_path, 'fold' + str(row['fold']), row['slice_file_name'])
                class_label = row['class']
                data = extract_features(final_path)
                extracted_features.append([data.tolist(), class_label])

            audio_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])
            os.makedirs(os.path.dirname(self.transformation_config.transformed_data_path), exist_ok=True)
            audio_df.to_csv(self.transformation_config.transformed_data_path, index=False)
            
            logging.info('Data Transformation Completed')

            return self.transformation_config.transformed_data_path

        except Exception as e:
            raise CustomException(e, sys)
