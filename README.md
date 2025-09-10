## Flight Price Prediction using AWS SageMaker

An end-to-end Machine Learning project implemented on AWS SageMaker, demonstrating the full lifecycle of a data science workflow — from raw data to a deployed ML web application.

This repository contains my personal implementation of the project, completed while learning and applying cloud-based ML engineering skills. Unlike a toy example, this project is built with production readiness in mind, leveraging AWS services and MLOps best practices.

## 🌐 Live Demo  

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-brightgreen?logo=streamlit)](https://aws-ml-flight-price-prediction-1-dj28ywh5gopajxzkdybgapp.streamlit.app/)


## Project Overview

The goal of this project is to predict flight ticket prices based on various factors such as airline, departure/arrival time, duration, source, and destination.

##### What makes this project stand out:

-End-to-End Pipeline → Data ingestion, preprocessing, feature engineering, model training, deployment, and monitoring.

-Cloud-Native → Entire ML workflow managed with AWS SageMaker, S3, IAM, and EC2.

- Production Deployment → A Streamlit web app hosted on cloud, making predictions accessible via an intuitive interface.

-Scalable & Reproducible → Modularized code with pipelines (scikit-learn transformers, SageMaker processing jobs).

## Project Workflow

#### Data Preparation

-Collected and cleaned dataset using Pandas & NumPy

-Handled missing values, outliers, categorical encoding, and feature scaling

-Exploratory Data Analysis (EDA)

-Conducted statistical analysis & hypothesis testing

-Generated visualizations (univariate, bivariate, multivariate) to uncover insights

#### Feature Engineering & Preprocessing

-Created custom scikit-learn transformers for categorical and numerical features

-Built preprocessing pipelines with Pipeline, FeatureUnion, ColumnTransformer

-Model Training & Optimization

-Performed Feature Selection to identify the most predictive variables.

-Trained  regression model (XGBoost) on AWS SageMaker

-Hyperparameter tuning using SageMaker’s built-in capabilities and also using Optuna in local environment

#### Model Deployment

-Saved model artifacts to S3

-Integrated with a Streamlit Web App for real-time predictions

#### Web Application

-Built a Streamlit-based UI for interactive predictions

-Users can enter flight details (airline, date, duration, etc.) and instantly get price predictions

-Deployed app on the cloud for public access

## Tech Stack

-Languages & Libraries: Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, XGBoost

-AWS Services: SageMaker, S3, EC2, IAM

-MLOps Tools: SageMaker Pipelines, Hyperparameter Tuning Jobs and Optuna

Deployment: Streamlit Web App

## Repository Structure

📦 flight-price-prediction
 ┣ 📂 data/                 # Raw and processed datasets
 ┣ 📂 notebooks/            # Jupyter notebooks for EDA & experimentation
 ┣ 📂 app.py                # Streamlit app & deployment scripts
 ┣ README.md                # Project documentation
 ┗ requirements.txt

### Getting Started

To explore this project, simply clone the repository and follow the setup steps:

git clone https://github.com/A-Hassan-72/AWS-ML-flight-price-prediction-1.git
cd AWS-ML-flight-price-prediction-1


##### Install the dependencies:

pip install -r requirements.txt


##### Run the Streamlit app locally:

streamlit run app.py