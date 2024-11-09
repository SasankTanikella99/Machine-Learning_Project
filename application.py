from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import StandardScaler

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
class PredictionInput(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: float
    writing_score: float

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
async def predict_form(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

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


