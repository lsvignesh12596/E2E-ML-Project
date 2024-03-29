import pandas as pd
import os, sys
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.Data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

from dataclasses import dataclass
@dataclass
class DataIngestionConfig():
    train_data_path:str = os.path.join("artifacts",'train.csv')
    test_data_path:str = os.path.join("artifacts", 'test.csv')
    raw_data_path:str = os.path.join("artifacts", 'raw_data.csv')

class DataIngestion():

    def __init__(self):
        self.ingestion_config = DataIngestionConfig

    def initiate_data_ingestion(self):
        logging.info("Entered Data Ingestion")
        try:
            # df = pd.read_csv(r'D:\Vignesh\GitUpload\E2E-ML-Project\notebook\data\stud.csv')
            df = pd.read_csv(r'notebook\data\stud.csv')
            logging.info('Read the input dataset')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train Test Split Initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    data_transform_obj = DataTransformation()
    train_arr, test_arr, _ = data_transform_obj.intitiate_data_transformation(train_path, test_path)

    modelTrainer = ModelTrainer()
    r2_score_score = modelTrainer.initiate_model_trainer(train_arr, test_arr, "")

    print("The best r2 score is ", r2_score_score)
