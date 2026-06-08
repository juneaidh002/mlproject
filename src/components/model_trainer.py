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
                "CatBoosting Refression" : CatBoostRegressor(verbose=True),
                "AdaBoost Classifier" : AdaBoostRegressor()
            }


            param_grids = {

                "Random Forest": {
                    "n_estimators": [100, 200, 500],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", None]
                },

                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                    "max_depth": [None, 5, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", None]
                },

                "Gradient Boosting": {
                    "n_estimators": [100, 200, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "min_samples_split": [2, 5, 10],
                    "subsample": [0.8, 0.9, 1.0]
                },

                "Linear Regression": {
                    # No major hyperparameters to tune
                    # Can use fit_intercept and positive
                    "fit_intercept": [True, False],
                    "positive": [True, False]
                },

                "K-Neighbors Regression": {
                    "n_neighbors": [3, 5, 7, 9, 11, 15],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                    "p": [1, 2]  # Manhattan, Euclidean
                },

                "XGBoosting Regression": {
                    "n_estimators": [100, 200, 500],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7, 10],
                    "subsample": [0.8, 0.9, 1.0],
                    "colsample_bytree": [0.8, 0.9, 1.0],
                    "gamma": [0, 0.1, 0.3],
                    "reg_alpha": [0, 0.01, 0.1],
                    "reg_lambda": [1, 1.5, 2]
                },

                "CatBoosting Regression": {
                    "iterations": [100, 300, 500],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "depth": [4, 6, 8, 10],
                    "l2_leaf_reg": [1, 3, 5, 7],
                    "border_count": [32, 64, 128]
                },

                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 1.0],
                    "loss": ["linear", "square", "exponential"]
                }
            }
            model_report:dict = evaluate_models(X_train=x_train,y_train=y_train,X_test = x_test,
                                               y_test = y_test,models=models)
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[ 
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

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