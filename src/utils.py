import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        trained_models = {}

        for i in range(len(list(models))):

            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = RandomizedSearchCV(
                model,
                para,
                n_iter=20,
                cv=3,
                n_jobs=-1,
                random_state=42
            )

            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_test_pred = best_model.predict(X_test)

            report[list(models.keys())[i]] = r2_score(y_test, y_test_pred)

            trained_models[list(models.keys())[i]] = best_model

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
