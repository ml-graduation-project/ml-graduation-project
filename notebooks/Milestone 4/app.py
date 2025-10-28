from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import uvicorn
from fastapi.responses import HTMLResponse
import pandas as pd
import io


model = load_model("model/lstmmodel.keras")
scaler = joblib.load("model/scaler.pkl")

app = FastAPI(
    title="Predictive Maintenance API (LSTM)",
    description="Predict Remaining Useful Life (RUL) from time-series sensor data",
    version="1.0"
)


class SensorData(BaseModel):

    sequence: list[list[float]]

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Predictive Maintenance API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    text-align: center;
                    background-color: #f8fafc;
                    color: #333;
                    margin-top: 100px;
                }
                h1 { color: #0078d7; }
                p { font-size: 18px; }
                a {
                    display: inline-block;
                    margin-top: 20px;
                    text-decoration: none;
                    background: #0078d7;
                    color: white;
                    padding: 10px 20px;
                    border-radius: 8px;
                }
                a:hover { background: #005fa3; }
            </style>
        </head>
        <body>
            <h1>Predictive Maintenance API (LSTM)</h1>
            <p>Use this API to predict Remaining Useful Life (RUL) from sensor data.</p>
            <a href="/docs">Go to API Docs</a>
        </body>
    </html>
    """

@app.post("/predict")
def predict_rul(data: SensorData):
    sequence = np.array(data.sequence)
    if sequence.shape[0] < 50:
        padding = np.zeros((50 - sequence.shape[0], sequence.shape[1]))
        sequence = np.vstack((padding, sequence))
    sequence_scaled = scaler.transform(sequence)
    sequence_scaled = sequence_scaled.reshape(1, 50, 17)
    prediction = model.predict(sequence_scaled)
    predicted_rul = float(prediction[0][0])

    return {"Predicted_RUL": predicted_rul}




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
