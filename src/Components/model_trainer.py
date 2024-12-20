import sys, os
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

try:
    from src.exception import customExceptionHandler
    from src.logger import logging
    from src.utils import save_obj, evaluate_models
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Project root being used: {PROJECT_ROOT}")
    raise

@dataclass
# The class `ModelTrainer_Config` contains a configuration setting for the file path where the trained
# model will be saved.
class ModelTrainer_Config:
    train_model_file_path = os.path.join("artifacts_output", "model.pkl")

class ModelTrainer:
    def __init__(self):
        """
        The `__init__` function initializes an instance of a class with a `model_trainer_config`
        attribute set to an instance of the `ModelTrainer_Config` class.
        """
        self.model_trainer_config = ModelTrainer_Config()

    def initiate_model_trainer(self, train_array, test_array):
        """
        The function `initiate_model_trainer` trains multiple regression models, evaluates their
        performance, selects the best model based on a threshold score, saves the best model, and
        returns the R2 score of the predictions.
        
        :param train_array: The `initiate_model_trainer` function you provided seems to be a part of a
        machine learning model training process. It takes `train_array` and `test_array` as input data
        for training and testing the models
        :param test_array: The `test_array` parameter in the `initiate_model_trainer` function is used
        to hold the test data for evaluating the trained models. It is expected to be a 2D numpy array
        where each row represents a sample and each column represents a feature or the target variable.
        The last column
        :return: The function `initiate_model_trainer` returns the R2 score calculated based on the best
        model found during training.
        """
        try:
            # Check for valid train_array and test_array
            if train_array is None or test_array is None:
                raise ValueError("train_array or test_array is None. Ensure data is loaded properly.")
            if train_array.shape[1] < 2 or test_array.shape[1] < 2:
                raise ValueError("train_array and test_array must have at least two columns (features and target).")

            logging.info("Splitting train and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Nearest Neighbors": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "XGBoost": XGBRFRegressor(),
                "CatBoosting": CatBoostRegressor(verbose=False),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            # Ensure evaluate_models returns a valid dictionary
            model_report = evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models, params=params)
        

            # Get the best model based on the score
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.7:
                raise customExceptionHandler("No best model found. Best model score is below 0.7 threshold.")
            
            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")

            # Save the best model
            save_obj(file_path=self.model_trainer_config.train_model_file_path, obj=best_model)

            # Predict and calculate R2 score
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            logging.error(f"Error occurred while training models: {e}")
            raise customExceptionHandler(e, sys)

