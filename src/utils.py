import os, sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

try:
    from src.exception import customExceptionHandler
    from src.logger import logging
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Project root being used: {PROJECT_ROOT}")
    raise

def save_obj(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        logging.error("Error in saving object", exc_info=True)
        raise customExceptionHandler(e) from None


def evaluate_models(x_train, y_train, x_test, y_test, models):
    try:
        results = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(x_train, y_train)

            # Make predictions
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            results[list(models.keys())[i]] = test_model_score
        return results
    
    except Exception as e:
        logging.error("Error in evaluating models", exc_info=True)
        raise customExceptionHandler(e,sys)

