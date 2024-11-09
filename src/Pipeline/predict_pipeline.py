import sys, os
import pandas as pd
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

try:
    from src.exception import customExceptionHandler
    from src.logger import logging
    from src.utils import load_obj
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Project root being used: {PROJECT_ROOT}")
    raise

class PredictionPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path="artifacts_output/model.pkl"
            preprocessor_path="artifacts_output/preprocessor.pkl"
            model=load_obj(file_path=model_path)
            preprocessor=load_obj(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            predictions=model.predict(data_scaled)
            return predictions
        
        except Exception as e:
            logging.error("Error in predicting", exc_info=True)
            raise customExceptionHandler(e) from None


class CustomData:  #mappping all inputs in html to backend
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education, lunch:str, test_preparation_course: str, reading_score: int, writing_score:str):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def get_data_as_df(self):
        try:
            custom_input_data = {
                'gender': [self.gender],
                'race_ethnicity': [self.race_ethnicity],
                'parental_level_of_education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test_preparation_course': [self.test_preparation_course],
                'reading_score': [self.reading_score],
                'writing_score': [self.writing_score]
            }
            return pd.DataFrame(custom_input_data)
        
        except Exception as e:
            logging.error(f"Error occurred while creating DataFrame: {e}")
            raise customExceptionHandler(f"Error occurred while creating DataFrame: {e}")