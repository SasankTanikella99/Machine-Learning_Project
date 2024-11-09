# Student Performance in Exams - End-to-End ML Project

This project is an end-to-end machine learning application that predicts student performance in exams. The dataset used for this project is sourced from Kaggle: [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977).

The project consists of:
- Data preprocessing and model training.
- A FastAPI web application for predicting student performance.
- Deployment on AWS Elastic Beanstalk.
- Continuous Deployment (CD) using AWS CodePipeline.



## Project Overview
This project predicts student performance based on various features such as gender, race/ethnicity, parental level of education, lunch type, and test preparation course completion. The model is trained using a classification algorithm and deployed as an API using FastAPI.

### Key Features:
- **Data Preprocessing**: Clean and preprocess the Kaggle dataset.
- **Model Training**: Train a machine learning model to predict student scores.
- **FastAPI Web App**: Serve predictions via a REST API.
- **AWS Elastic Beanstalk Deployment**: Deploy the FastAPI app on AWS Elastic Beanstalk.
- **CI/CD Pipeline**: Automate deployment with AWS CodePipeline.

## Technologies Used
- **Python 3.12**
- **FastAPI**
- **Uvicorn**
- **Scikit-learn**
- **Docker**
- **AWS Elastic Beanstalk**
- **AWS CodePipeline**

## Algorithms
1. **CatBoost Regressor**
2. **AdaBoost Regressor**
3. **Gradient Boosting Regressor**
4. **Random Forest Regressor**
5. **Linear Regression**
6. **K-Nearest Neighbors (KNN) Regressor**
7. **Decision Tree Regressor**
8. **XGBoost Regressor**

## Prerequisites
Before running or deploying this project, ensure you have the following installed:
- Python (>= 3.8)
- Docker
- AWS CLI
- AWS Elastic Beanstalk CLI (EB CLI)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/student-performance-prediction.git
cd student-performance-prediction

## 2. Install Dependencies

### Create a virtual environment:
```bash
conda create -p  ./venv python==3.12 -y  
```

### Activate the virtual environment:

- :
    ```bash
  conda activate ./venv
    ```

### Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 3. Train the Model

Run the script:
```bash
python src/Components/data_ingestion.py
```

## 4. Run the Application Locally

To run the FastAPI app locally using Uvicorn:
```bash
uvicorn application:app --reload --host 0.0.0.0 --port 8000
```

### Output
![Screenshot 2024-11-09 at 11 51 20 AM](https://github.com/user-attachments/assets/afb76c4d-619b-4173-8ec8-1491e1a46f35)
![Screenshot 2024-11-09 at 11 51 10 AM](https://github.com/user-attachments/assets/d9289e6b-a66a-4a87-8528-d830d0a8c54b)
![image](https://github.com/user-attachments/assets/3a8900f9-9e47-4624-b58c-d453d4b3ea6f)
![image](https://github.com/user-attachments/assets/198af6d4-5424-46d6-99c4-00a07ffcba07)

