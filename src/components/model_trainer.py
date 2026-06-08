import os
import sys

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regression" : KNeighborsRegressor(),
                "XGBoosting Regression" : XGBRegressor(),
                "CatBoosting Regression" : CatBoostRegressor(verbose=True),
                "AdaBoost Regressor" : AdaBoostRegressor()
            }


            param_grids = {

                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20],
                    "min_samples_split": [2, 5]
                },

                "Decision Tree": {
                    "max_depth": [5, 10, 20],
                    "min_samples_split": [2, 5]
                },

                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5]
                },

                "Linear Regression": {
                    "fit_intercept": [True, False]
                },

                "K-Neighbors Regression": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"]
                },

                "XGBoosting Regression": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0]
                },

                "CatBoosting Regression": {
                    "iterations": [100, 300],
                    "learning_rate": [0.05, 0.1],
                    "depth": [4, 6, 8]
                },

                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.05, 0.1]
                }
            }
            model_report, trained_models = evaluate_models(
                X_train=x_train,
                y_train=y_train,
                X_test=x_test,
                y_test=y_test,
                models=models,
                param=param_grids
                )           
            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[ 
                list(model_report.values()).index(best_model_score)
            ]

            best_model = trained_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(x_test)

            r2_square = r2_score(y_test,predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)