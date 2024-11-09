
# Import necessary libraries
import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# The code snippet you provided is setting up the project root path and adding it to the system path
# in order to import local modules from the project structure. Here's a breakdown of what each part is
# doing:
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

try:
    from src.exception import customExceptionHandler
    from src.logger import logging
    from src.utils import save_obj
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Project root being used: {PROJECT_ROOT}")
    raise

# The class `DataTransformation_Config` contains a file path for a preprocessor object.
class DataTransformation_Config:
    preprocessor_object_file_path=os.path.join('artifacts_output', 'preprocessor.pkl')

class DataTransformation:

    """
        The `__init__` function initializes an instance of the class with a `data_transformation_config`
        attribute set to an instance of `DataTransformation_Config`.
    
    """
    def __init__(self):
        self.data_transformation_config=DataTransformation_Config()
    

    """
        The function `get_data_tranformer_object` creates a data transformer object for processing
        numerical and categorical columns in a dataset.
        :return: A `ColumnTransformer` object is being returned, which contains two pipelines for
        processing numerical and categorical columns in a dataset. The numerical pipeline includes steps
        for handling missing values with median imputation and standard scaling, while the categorical
        pipeline includes steps for handling missing values with mode imputation, one-hot encoding, and
        standard scaling.
    """
    def get_data_tranformer_object(self):
        
        try:
            numerical_column_data = ["writing_score", "reading_score"]
            categorical_column_data = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),    # handling missing values
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')), # replace missing values with mode
                ('encoder', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info("Numerical columns standard scaling is completed")

            logging.info("Categorical columns encoding is completed")

            preprocessor = ColumnTransformer([
                    ('numerical_pipeline', numerical_pipeline, numerical_column_data),
                    ('categorical_pipeline', categorical_pipeline, categorical_column_data)
                ])
            
            return preprocessor

        except Exception as e:
            logging.error("Error in getting data transformer object: ", exc_info=True)
            raise customExceptionHandler(e)


    """
        The function `initiate_data_transformer` reads train and test data, prepares the data, creates a
        data transformer object, and saves the preprocessor object.
        
        :param train_path: The `train_path` parameter is the file path to the training data CSV file
        :param test_path: The `test_path` parameter in the `initiate_data_transformer` function is the
        file path where the test data is stored. This function reads the test data from the specified
        file path to perform data transformation tasks
        :return: (train_arr, test_arr, self.data_transformation_config.preprocessor_object_file_path)
    """

    def initiate_data_transformer(self,train_path,test_path):
        
        try:
            train_dataf = pd.read_csv(train_path)
            test_dataf = pd.read_csv(test_path)

            logging.info("Data loading(reading test and train data) is completed")

            logging.info("obtaining preprocessor object")

            preprocessor_object = self.get_data_tranformer_object()

            target_column_data = "math_score"
            numerical_column_data=["writing_score", "reading_score"]

            input_feature_train_dataf=train_dataf.drop(columns=[target_column_data], axis=1)
            target_feature_train_dataf=train_dataf[target_column_data]

            logging.info("Training data preparation is completed")


            input_feature_test_dataf=test_dataf.drop(columns=[target_column_data], axis=1)
            target_feature_test_dataf=test_dataf[target_column_data]

            logging.info("Test data preparation is completed")

            input_feature_train_arr=preprocessor_object.fit_transform(input_feature_train_dataf)
            input_feature_test_arr=preprocessor_object.transform(input_feature_test_dataf)

            logging.info("Data transformer object created")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_dataf)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_dataf)]

            logging.info("Training and test data is prepared")

            save_obj(file_path=self.data_transformation_config.preprocessor_object_file_path, obj=preprocessor_object)

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_object_file_path)
        
        except Exception as e:
            logging.error("Error in initiating data transformer: ", exc_info=True)
            raise customExceptionHandler(e)
