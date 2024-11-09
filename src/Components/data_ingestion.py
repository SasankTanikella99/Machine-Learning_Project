import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

# The provided Python script defines a `DataIngestion` class that handles the process of ingesting
# data, including reading a dataset, saving raw data, performing train-test split, and saving train
# and test sets.
# The code block you provided is fixing the path resolution for local imports in a Python script.
# Here's what it does:
# Fix the path resolution for local imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

try:
    from src.exception import customExceptionHandler
    from src.logger import logging
    from src.utils import save_obj
    from src.Components.data_transformation import DataTransformation_Config
    from src.Components.data_transformation import DataTransformation

except ImportError as e:
    print(f"Error importing local modules: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Project root being used: {PROJECT_ROOT}")
    raise

# The `@dataclass` decorator in Python is used to automatically generate special methods such as
# `__init__`, `__repr__`, `__eq__`, and `__hash__` for a class.
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts_output', "train.csv")
    test_data_path: str = os.path.join('artifacts_output', "test.csv")
    raw_data_path: str = os.path.join('artifacts_output', "data.csv")

# The `DataIngestion` class handles the process of ingesting data, including reading a dataset, saving
# raw data, performing train-test split, and saving train and test sets.

class DataIngestion:
    def __init__(self):
        """
        The function initializes an instance variable for data ingestion configuration.
        """
        self.ingestion_config = DataIngestionConfig()
    
    def initiating_data_ingestion(self):
        logging.info("Starting data ingestion component")
        try:
            # Create artifacts_output directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Read the dataset
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Reading the dataset as a dataframe")

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Saved raw data to CSV")

            # Perform train-test split
            logging.info("Starting Train Test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=38)

            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )
        
        except Exception as e:
            logging.error("Error in data ingestion", exc_info=True)
            raise customExceptionHandler(e) from None

# The `if __name__ == "__main__":` block in Python is used to check whether the script is being run
# directly by the Python interpreter or if it is being imported as a module into another script.
if __name__ == "__main__":
    try:
        obj = DataIngestion()
        train_data, test_data, raw_data = obj.initiating_data_ingestion()

        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformer(train_data, test_data)
        
    except Exception as e:
        print(f"An error occurred: {e}")