# AI-Powered Predictive Maintenance for Industrial Equipment

## Overview

This project leverages machine learning and IoT sensor data (including vibration, temperature, pressure, and humidity) to predict equipment failures before they occur.
By forecasting issues early, the system helps industries reduce downtime, optimize maintenance schedules, and improve operational efficiency.

## Objectives

- Predict equipment failures in advance using historical sensor data.

- Minimize unexpected breakdowns and associated costs.

- Implement intelligent maintenance scheduling based on data-driven insights.

- Deploy a real-time monitoring system with continuous model updates.

## System Architecture

1- IoT Sensors – Collect real-time equipment data (temperature, vibration, etc.)

2- Data Preprocessing Layer – Cleans, and normalizes data 

3-Machine Learning Model (LSTM) – Learns temporal patterns for failure prediction

4- API/Deployment Layer – Provides prediction endpoints via Streamlit

## Dataset

Source: Public industrial sensor datasets (e.g., NASA Turbofan Engine Degradation Simulation Dataset / CMAPSS)

Link: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

Features: Vibration, pressure, temperature, humidity, operating conditions

Target: Remaining Useful Life (RUL) or failure event indicator

## Data Preprocessing

- Handling missing values

- Normalization and scaling

- Sequence creation for LSTM (time-series windowing)

- Train-test split by machine/unit

## Machine Learning Models

| Model             | Description                                       | Purpose    |
| ----------------- | ------------------------------------------------- | ---------- |
| **LSTM**          | Captures temporal dependencies in sensor readings | Main model |
| **Random Forest** | Baseline traditional ML model                     | Benchmark  |
| **XGBoost**       | Gradient boosting for performance comparison      | Benchmark  |

## Model Evaluation

Metrics: R2 score,root mean squared error

LSTM Results: 
Train Results --> RMSE: 12.16, R²: 0.914
Test Results --> RMSE: 15.08, R²: 0.88

Visualizations: prediction vs actual degradation plots

## Tech Stack

- Python

- TensorFlow / Keras

- Scikit-learn

- Pandas, NumPy, Matplotlib, Seaborn

- Streamlit

## Results & Insights

- Early fault prediction helps schedule maintenance efficiently.

- LSTM achieved higher accuracy in detecting degradation trends.

- Reduced downtime and maintenance cost in simulated environment.

## Future Improvements

- Incorporate transformer-based models for multivariate time-series.

- Integrate edge computing for on-device predictions.

- Expand dataset with more diverse sensor types.
