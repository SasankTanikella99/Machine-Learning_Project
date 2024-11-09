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
class ModelTrainer_Config:
    train_model_file_path = os.path.join("artifacts_output", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainer_Config()

    def initiate_model_trainer(self, train_array, test_array):
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

            # Ensure evaluate_models returns a valid dictionary
            model_report = evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models)
        

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
            raise customExceptionHandler(f"An error occurred in ModelTrainer: {str(e)}")

