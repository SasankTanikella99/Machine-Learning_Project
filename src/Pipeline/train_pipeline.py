import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

try:
    from src.exception import customExceptionHandler
    from src.logger import logging
    from src.utils import load_obj,save_obj
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Project root being used: {PROJECT_ROOT}")
    raise

class TrainPipeline:
    def __init__(self):
        self.artifacts_path = "artifacts"
        os.makedirs(self.artifacts_path, exist_ok=True)
        self.model_path = os.path.join(self.artifacts_path, "model.pkl")
        self.preprocessor_path = os.path.join(self.artifacts_path, "preprocessor.pkl")
        self.models = {
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "AdaBoost": AdaBoostRegressor(),
            "Linear Regression": LinearRegression(),
        }

    def load_data(self, file_path):
        """
        Load and preprocess data from a CSV file.
        """
        try:
            data = pd.read_csv(file_path)
            X = data.drop("target_column", axis=1)  # Replace "target_column" with your actual target column
            y = data["target_column"]
            return X, y
        except Exception as e:
            raise customExceptionHandler(f"Error loading data: {e}", sys)

    def preprocess_data(self, X_train, X_test):
        """
        Scale the features using StandardScaler and save the scaler.
        """
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            save_obj(self.preprocessor_path, scaler)
            logging.info("Data preprocessing and scaling complete.")
            return X_train_scaled, X_test_scaled
        except Exception as e:
            raise customExceptionHandler(f"Error in data preprocessing: {e}", sys)

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """
        Train multiple models, evaluate them, and save the best-performing model.
        """
        try:
            best_score = -1
            best_model_name = None
            best_model = None

            for model_name, model in self.models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)
                logging.info(f"{model_name} R2 score: {score}")

                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    best_model = model

            if best_score < 0.7:  # Assuming 0.7 is your threshold for an acceptable model
                raise customExceptionHandler("No model met the minimum R2 score requirement.", sys)

            save_obj(self.model_path, best_model)
            logging.info(f"Best model '{best_model_name}' with R2 score {best_score} saved successfully.")

            return best_model_name, best_score
        except Exception as e:
            raise customExceptionHandler(f"Error in model training and evaluation: {e}", sys)

    def run_training_pipeline(self, data_file_path):
        """
        Complete training pipeline to load data, preprocess, train, evaluate, and save the model.
        """
        try:
            # Load and split data
            X, y = self.load_data(data_file_path)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Preprocess data
            X_train_scaled, X_test_scaled = self.preprocess_data(X_train, X_test)
            
            # Train and evaluate models
            best_model_name, best_score = self.train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test)
            
            logging.info(f"Training pipeline completed. Best model: {best_model_name}, R2 Score: {best_score}")
            return best_model_name, best_score
        except Exception as e:
            raise customExceptionHandler(f"Error in training pipeline: {e}", sys)

