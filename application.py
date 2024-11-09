from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import StandardScaler

# The code snippet you provided is performing the following actions:
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

try:
    from src.exception import customExceptionHandler
    from src.logger import logging
    from src.Pipeline.predict_pipeline import CustomData, PredictionPipeline
    from src.Pipeline.train_pipeline import TrainPipeline 
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Project root being used: {PROJECT_ROOT}")

# Initialize FastAPI
app = FastAPI()

# Set up Jinja2 templates for HTML rendering
templates = Jinja2Templates(directory="templates")

# Pydantic model for prediction input
# This Python class named PredictionInput defines attributes related to student demographics and
# academic performance for making predictions.
class PredictionInput(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: float
    writing_score: float

"""
    This Python function serves an HTML template named "index.html" when the root URL is accessed.
    
    :param request: The `request` parameter in the `index` function is of type `Request`, which
    represents an incoming HTTP request. It contains information about the request such as headers,
    cookies, query parameters, and more. In this case, it is used to pass the request object to the
    template `index.html
    :type request: Request
    :return: An HTML response is being returned, using the "index.html" template and passing the request
    object to the template.
"""

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


"""
    This Python function serves an HTML form for making predictions.
    
    :param request: The `request` parameter in the `predict_form` function represents the incoming HTTP
    request made to the server. It contains information such as the request method, headers, cookies,
    query parameters, and more. In this case, it is being passed to the `predict_form` function to
    generate a response
    :type request: Request
    :return: The code is returning a HTML response using a template called "home.html" along with the
    request object.
"""

@app.get("/predict", response_class=HTMLResponse)
async def predict_form(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

"""
    This Python function defines an endpoint `/predict` that takes input data from a form, processes it,
    makes a prediction using a prediction pipeline, and renders the results in an HTML template.
    
    :param request: The `request` parameter in the `predict` function represents the incoming HTTP
    request made to the `/predict` endpoint. It contains information about the request such as headers,
    cookies, query parameters, etc., and allows you to access and interact with the request data
    :type request: Request
    :param gender: The `gender` parameter in the code snippet represents the gender of a student. It is
    a required field that is expected to be provided as a form input when making a POST request to the
    `/predict` endpoint. The gender can be specified as a string value
    :type gender: str
    :param ethnicity: The `ethnicity` parameter in the code snippet represents the race or ethnicity of
    the student. This information is used as one of the features for predicting the outcome in the
    model. It is passed as a form parameter in the POST request to the `/predict` endpoint
    :type ethnicity: str
    :param parental_level_of_education: The `parental_level_of_education` parameter in the code snippet
    you provided is a form field that expects a string input. This parameter is used to capture the
    parental level of education of a student as part of the data required for making a prediction
    :type parental_level_of_education: str
    :param lunch: The `lunch` parameter in the code snippet you provided is a form field that likely
    represents the type of lunch the student has. It is a string field that is expected to be provided
    as part of the form data when making a POST request to the `/predict` endpoint. The lunch options
    could
    :type lunch: str
    :param test_preparation_course: The `test_preparation_course` parameter in the code snippet you
    provided is a form field that expects a string input. It likely represents whether or not the
    student has completed a test preparation course. This information is used as one of the features for
    predicting the outcome in the prediction pipeline
    :type test_preparation_course: str
    :param reading_score: The `reading_score` parameter in the code snippet represents the score
    obtained in the reading section of a test. It is a float value that is provided as a form input when
    making a POST request to the `/predict` endpoint. This score is used as one of the features for
    predicting an outcome using
    :type reading_score: float
    :param writing_score: The `writing_score` parameter in the code snippet represents the score
    obtained in a writing test. This parameter is expected to be provided as a float value in the form
    data when making a POST request to the `/predict` endpoint. The code snippet processes this input
    along with other parameters to generate a prediction
    :type writing_score: float
    :return: The code is returning a HTML response using the "home.html" template with the prediction
    results displayed on the page.
"""

@app.post("/predict", response_class=HTMLResponse)
   
async def predict(
    request: Request,
    gender: str = Form(...),
    ethnicity: str = Form(...),
    parental_level_of_education: str = Form(...),
    lunch: str = Form(...),
    test_preparation_course: str = Form(...),
    reading_score: float = Form(...),
    writing_score: float = Form(...)
):
    data = CustomData(
        gender=gender,
        race_ethnicity=ethnicity,
        parental_level_of_education=parental_level_of_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        reading_score=reading_score,
        writing_score=writing_score
    )

    # Convert data to DataFrame
    prediction_df = data.get_data_as_df()
    print(prediction_df)

    # Predict
    predict_pipeline = PredictionPipeline()
    results = predict_pipeline.predict(prediction_df)

    # Render template with results
    return templates.TemplateResponse("home.html", {"request": request, "results": results[0]})


"""
    The function `/train` in this Python code snippet handles a POST request to train a machine learning
    model using a specified data file path and returns the best model name and score upon successful
    completion.
    
    :param data_file_path: The `data_file_path` parameter in the `/train` endpoint is a required
    parameter that should be provided as a form field when making a POST request to the endpoint. This
    parameter should contain the file path to the data that will be used for training the model. If this
    parameter is not provided or
    :type data_file_path: str
    :return: The endpoint `/train` is returning a JSON response with a message indicating whether the
    training was completed successfully or not. If successful, it includes the best model name and the
    best score achieved during training. If an exception occurs during training, an error message is
    logged and a 500 status code response is returned with details of the error.
"""

@app.post("/train", response_class=JSONResponse)
    
async def train(data_file_path: str = Form(...)):
    if not data_file_path:
        raise HTTPException(status_code=400, detail="Data file path is required")

    try:
        train_pipeline = TrainPipeline()
        best_model_name, best_score = train_pipeline.run_training_pipeline(data_file_path)
        return {"message": "Training completed successfully", "best_model": best_model_name, "best_score": best_score}
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


