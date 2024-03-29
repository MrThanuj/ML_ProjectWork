import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataPreprocessor
from src.components.data_transformation import PreprocessingConfig

from src.components.model_trainer import ModelTrainingConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class IngestionConfig:
    path_to_training_data: str = os.path.join('artifacts', "train.csv")
    path_to_testing_data: str = os.path.join('artifacts', "test.csv")
    path_to_raw_data: str = os.path.join('artifacts', "data.csv")

class DataIngest:
    def __init__(self):
        self.config = IngestionConfig()

    def start_ingestion(self):
        logging.info("Beginning the data ingestion process")
        try:
            dataset = pd.read_csv('notebook\data\stud.csv')
            logging.info('Dataset loaded into a pandas data frame')

            os.makedirs(os.path.dirname(self.config.path_to_training_data), exist_ok=True)

            dataset.to_csv(self.config.path_to_raw_data, index=False, header=True)

            logging.info("Splitting data into training and testing sets")
            training_data, testing_data = train_test_split(dataset, test_size=0.2, random_state=42)

            training_data.to_csv(self.config.path_to_training_data, index=False, header=True)
            testing_data.to_csv(self.config.path_to_testing_data, index=False, header=True)

            logging.info("Data ingestion process completed")

            # In the provided code snippet, there is no reference to a variable named `ret`. It seems
            # like there might be a misunderstanding or missing context. If you can provide more
            # information or specify where `ret` is used in the code, I would be happy to help clarify
            # its purpose or functionality.
            return (
                self.config.path_to_training_data,
                self.config.path_to_testing_data
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    ingest = DataIngest()
    path_to_train, path_to_test = ingest.start_ingestion()

    data_transformation = DataPreprocessor()
    train_matrix, test_matrix, _ = data_transformation.initiate_data_transformation(path_to_train, path_to_test)

    trainer = ModelTrainer()
    print(trainer.start_training(train_matrix, test_matrix))