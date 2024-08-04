import os
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_preparation import PrepareModel
from src.components.training import ModelTraining
from src.exception import CustomException
from src.logger import logging

class TrainPipeline:
    def __init__(self) -> None:
        pass

    def train(self):
        try:
            logging.info('TRAINING PIPELINE INITIATED')
            # ingestion = DataIngestion()
            # data_path = ingestion.initiate_data_ingestion()

            # data_path = os.path.join('artifacts', 'data')
            # transformation = DataTransformation()
            # transformed_path = transformation.initiate_data_transformation(data_path)

            transformed_path = os.path.join('artifacts', 'transformed_data', 'transformed_data.csv')

            prepare = PrepareModel()
            model = prepare.initiate_model_preparation()

            training = ModelTraining()
            trained_model = training.initiate_training(transformed_path, model)
            
            logging.info('TRAINING PIPELINE COMPLETED')

            return trained_model
            
        
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__=='__main__':
    train = TrainPipeline()
    train.train()