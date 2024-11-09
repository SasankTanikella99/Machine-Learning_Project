import os, sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

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

"""
    The `save_obj` function saves an object to a file using dill serialization, handling exceptions and
    logging errors.
    
    :param file_path: The `file_path` parameter in the `save_obj` function is the path where you want to
    save the object. It should be a string representing the file path including the file name and
    extension where you want to store the object data. For example, it could be something like
    "data/my_object
    :param obj: The `obj` parameter in the `save_obj` function is the object that you want to save to a
    file. This object can be of any data type that is serializable using the `dill` library's `dump`
    method. The `dill.dump(obj, file_obj)` line
"""
def save_obj(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        logging.error("Error in saving object", exc_info=True)
        raise customExceptionHandler(e) from None



"""
        The function `load_obj` loads an object from a file using dill and handles exceptions by logging
        errors and raising a custom exception.
        
        :param file_path: The `file_path` parameter in the `load_obj` function is a string that represents
        the path to the file from which an object needs to be loaded. This function attempts to open the
        file in binary mode, load the object using the `dill.load` method, and return the loaded object
        :return: The function `load_obj` is returning the object loaded from the file specified by the
        `file_path`. The object is loaded using the `dill.load` method within a `with` block that opens the
        file in binary read mode. If an exception occurs during the loading process, an error message is
        logged using the `logging.error` method, and a custom exception `customExceptionHandler` is raised
"""
def load_obj(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        logging.error("Error in loading object", exc_info=True)
        raise customExceptionHandler(e) from None


    """
    The function `evaluate_models` trains and evaluates multiple models using GridSearchCV and returns
    the test scores for each model.
    
    :param x_train: X_train is the training data features, which are used to train the machine learning
    models. It typically consists of input variables or features
    :param y_train: It seems like you were about to provide more information about the parameters used
    in the `evaluate_models` function. Could you please provide the rest of the information so that I
    can assist you further?
    :param x_test: It seems like your message got cut off. Could you please provide me with the rest of
    the information about the parameters so that I can assist you further?
    :param y_test: It seems like you were about to provide more information about the parameters, but
    the message got cut off. Could you please provide more details about the parameters y_test and any
    other information you would like to share?
    :param models: The `models` parameter in the `evaluate_models` function is a dictionary where the
    keys are model names or identifiers, and the values are the actual model objects. These model
    objects could be machine learning models like Linear Regression, Random Forest, etc
    :param params: It seems like you forgot to provide the details of the `params` variable. Could you
    please provide the details of the `params` variable so that I can assist you further with the
    `evaluate_models` function?
    :return: The function `evaluate_models` returns a dictionary containing the test R-squared scores
    for each model specified in the input `models` dictionary. The keys of the dictionary are the names
    of the models, and the values are the corresponding test R-squared scores.
    """
def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    try:
        results = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model,param,cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
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

