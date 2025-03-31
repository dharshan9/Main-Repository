import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load trained LSTM model and scaler
model = tf.keras.models.load_model("health_risk_model.h5")
scaler = joblib.load("scaler.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define request body model (fields match Streamlit)
class HealthInput(BaseModel):
    heart_rate: float
    sleep_duration: float
    timestamp_numeric: int
    sleep_category: int  # Unused but required for schema

@app.get("/")
def home():
    return {"message": "🏥 AI Health Tracker API is running!"}

@app.post("/predict")
def predict(data: HealthInput):
    try:
        # Create DataFrame with the correct feature names
        input_df = pd.DataFrame([[data.heart_rate, data.sleep_duration, data.timestamp_numeric]], 
                                columns=["heart_rate", "sleep_duration", "timestamp_numeric"])  # MATCH Streamlit JSON keys

        # Normalize input using the same scaler
        input_data = scaler.transform(input_df)

        # Reshape for LSTM (batch_size=1, time_steps=6, features=3)
        input_data = np.tile(input_data, (6, 1))  # Repeat input 6 times
        input_data = np.reshape(input_data, (1, 6, 3))

        # Make prediction
        prediction = model.predict(input_data)[0][0]
        risk_status = "At Risk" if prediction > 0.5 else "Healthy"

        return {"prediction": float(prediction), "status": risk_status}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
