import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class CustomData:
    """
    Prepares and structures raw input data for predictions.
    """
    def __init__(self, gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score):
        self.input_details = {
            "gender": gender,
            "race_ethnicity": race_ethnicity,
            "parental_level_of_education": parental_level_of_education,
            "lunch": lunch,
            "test_preparation_course": test_preparation_course,
            "reading_score": reading_score,
            "writing_score": writing_score,
        }

    def get_data_as_dataframe(self):
        """
        Converts input data to a DataFrame suitable for the prediction model.
        """
        try:
            return pd.DataFrame([self.input_details])
        except Exception as error:
            raise CustomException(f"Error converting input data to DataFrame: {error}", sys)

class PredictPipeline:
    """
    Facilitates the prediction process using a trained model and preprocessing pipeline.
    """
    def __init__(self):
        self.model_filepath = "artifacts/model.pkl"
        self.preprocessor_filepath = "artifacts/preprocessor.pkl"

    def predict(self, input_features):
        """
        Generates predictions for the given input features.
        """
        try:
            # Loading the necessary model and preprocessing tools
            model = load_object(self.model_filepath)
            preprocessor = load_object(self.preprocessor_filepath)

            # Applying preprocessing to the input features
            prepared_features = preprocessor.transform(input_features)
            # Making predictions with the processed features
            predictions = model.predict(prepared_features)

            return predictions
        except Exception as error:
            raise CustomException(f"Failed to make predictions: {error}", sys)