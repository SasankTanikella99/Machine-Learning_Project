Student Performance in Exams - End-to-End ML Project
This project is an end-to-end machine learning application that predicts student performance in exams. The dataset used for this project is sourced from Kaggle: Students Performance in Exams.
The project consists of:
Data preprocessing and model training.
A FastAPI web application for predicting student performance.
Deployment on AWS Elastic Beanstalk.
Continuous Deployment (CD) using AWS CodePipeline.
Table of Contents
Project Overview
Technologies Used
Prerequisites
Setup Instructions
1. Clone the Repository
2. Install Dependencies
3. Dataset
4. Train the Model
5. Run the Application Locally
6. Dockerize the Application
7. Deploy on AWS Elastic Beanstalk
Continuous Deployment with AWS CodePipeline
Screenshots
Contributing
License
Project Overview
This project predicts student performance based on various features such as gender, race/ethnicity, parental level of education, lunch type, and test preparation course completion. The model is trained using a classification algorithm and deployed as an API using FastAPI.
Key Features:
Data Preprocessing: Clean and preprocess the Kaggle dataset.
Model Training: Train a machine learning model to predict student scores.
FastAPI Web App: Serve predictions via a REST API.
AWS Elastic Beanstalk Deployment: Deploy the FastAPI app on AWS Elastic Beanstalk.
CI/CD Pipeline: Automate deployment with AWS CodePipeline.
Technologies Used
Python 3.12
FastAPI
Uvicorn
Scikit-learn
Pandas
Docker
AWS Elastic Beanstalk
AWS CodePipeline
Prerequisites
Before running or deploying this project, ensure you have the following installed:
Python (>= 3.8)
Docker
AWS CLI
AWS Elastic Beanstalk CLI (EB CLI)
Setup Instructions
1. Clone the Repository
bash
git clone https://github.com/yourusername/student-performance-prediction.git
cd student-performance-prediction

2. Install Dependencies
Create a virtual environment and install dependencies:
bash
python3 -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt

3. Dataset
Download the dataset from Kaggle:
Students Performance in Exams Dataset
Place the dataset (StudentsPerformance.csv) in the data/ directory.
4. Train the Model
Run the script to preprocess data and train your model:
bash
python train_model.py

This will save the trained model as model.pkl in the models/ directory.
5. Run the Application Locally
To run the FastAPI app locally using Uvicorn:
bash
uvicorn application:app --reload --host 0.0.0.0 --port 8000

Visit http://127.0.0.1:8000/docs to access the interactive API documentation (Swagger UI).
6. Dockerize the Application
To containerize your FastAPI app using Docker:
Build your Docker image:
bash
docker build -t student-performance-app .

Run your Docker container:
bash
docker run -d -p 8000:8000 student-performance-app

Access your app at http://localhost:8000.
7. Deploy on AWS Elastic Beanstalk
Initialize your Elastic Beanstalk environment:
bash
eb init -p python3.12 student-performance-app --region us-west-2

Create an Elastic Beanstalk environment and deploy:
bash
eb create student-performance-env
eb deploy

Once deployed, you can access your app via the generated Elastic Beanstalk URL.
Continuous Deployment with AWS CodePipeline
To automate deployments with AWS CodePipeline:
Set up a repository in GitHub or CodeCommit.
Configure a pipeline in AWS CodePipeline with stages for source control, build (using CodeBuild), and deployment (using Elastic Beanstalk).
Push changes to your repository to trigger automatic deployments.
For detailed instructions on setting up AWS CodePipeline, refer to the official documentation.
