import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def persist_data(target_path, data_object):
    """
    Persist a given Python object into a file located at target_path.
    """
    try:
        base_directory = os.path.dirname(target_path)
        os.makedirs(base_directory, exist_ok=True)
        with open(target_path, 'wb') as file_handler:
            pickle.dump(data_object, file_handler)
    except Exception as error:
        raise CustomException(error, sys)

def retrieve_data(source_path):
    """
    Retrieve a Python object stored at source_path.
    """
    try:
        with open(source_path, 'rb') as file_handler:
            return pickle.load(file_handler)
    except Exception as error:
        raise CustomException(error, sys)

def compare_model_performances(train_features, train_labels, test_features, test_labels, model_dict, params_dict):
    """
    Evaluate and compare multiple models based on the training and testing datasets.
    """
    evaluation_summary = {}
    try:
        for model_name, model_instance in model_dict.items():
            model_params = params_dict.get(model_name, {})
            grid_search = GridSearchCV(estimator=model_instance, param_grid=model_params, cv=5, scoring='r2')
            grid_search.fit(train_features, train_labels)

            optimal_model = grid_search.best_estimator_
            optimal_model.fit(train_features, train_labels)

            predictions_train = optimal_model.predict(train_features)
            predictions_test = optimal_model.predict(test_features)

            score_train = r2_score(train_labels, predictions_train)
            score_test = r2_score(test_labels, predictions_test)

            evaluation_summary[model_name] = score_test

        return evaluation_summary
    except Exception as error:
        raise CustomException(error, sys)