import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import extract_zip_file


@dataclass
class DataIngestionConfig:
    data_path: str = os.path.join('artifacts', 'data')


class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Initiated')
        try:
            extract_zip_file('C:/Users/SaraImdad/Desktop/data.zip', self.ingestion_config.data_path)
            logging.info('Data Ingestion Completed')

        except Exception as e:
            raise CustomException(e, sys)
        
        return self.ingestion_config.data_path
            



