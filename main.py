import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load Model and Scaler
model = tf.keras.models.load_model("health_risk_model.h5")
scaler = joblib.load("scaler.pkl")

# FastAPI app
app = FastAPI()

# Define request model
class HealthInput(BaseModel):
    heart_rate: float
    sleep_duration: float
    timestamp_numeric: int  # Ensure API gets 3 features

@app.get("/")
def home():
    return {"message": "🏥 AI Health Tracker API is running!"}

@app.post("/predict")
def predict(data: HealthInput):
    try:
        # Convert input to DataFrame with expected feature names
        input_df = pd.DataFrame([[data.heart_rate, data.sleep_duration, data.timestamp_numeric]], 
                                columns=["Heart Rate", "Sleep Duration", "Timestamp_Numeric"])
        
        # Normalize using the scaler
        input_data = scaler.transform(input_df)

        # Reshape for LSTM model
        input_data = np.reshape(input_data, (1, 1, 3))  # Match 3 features

        # Predict
        prediction = model.predict(input_data)[0][0]
        risk_status = "At Risk" if prediction > 0.5 else "Healthy"

        return {"prediction": float(prediction), "status": risk_status}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
