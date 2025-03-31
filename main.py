import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load trained LSTM model and scaler
try:
    model = tf.keras.models.load_model("health_risk_model.h5")
    scaler = joblib.load("scaler.pkl")
    logger.info("✅ Model & scaler loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load model/scaler: {e}")
    raise RuntimeError("Error loading model or scaler.")

# Initialize FastAPI app
app = FastAPI()

# Define request body model
class HealthInput(BaseModel):
    heart_rate: float
    sleep_duration: float
    timestamp_numeric: int

@app.get("/")
def home():
    return {"message": "🏥 AI Health Tracker API is running!"}

@app.post("/predict")
def predict(data: HealthInput):
    try:
        # Create DataFrame with correctly named columns
        input_df = pd.DataFrame([[data.heart_rate, data.sleep_duration, data.timestamp_numeric]], 
                                columns=["Heart Rate", "Sleep Duration", "Timestamp_Numeric"])  # FIXED column names
        
        # Normalize input using the saved scaler
        input_data = scaler.transform(input_df)

        # Ensure correct shape for LSTM (batch_size=1, time_steps=1, features=3)
        input_data = np.reshape(input_data, (1, 1, 3))

        # Make prediction
        prediction = model.predict(input_data)[0]  # Get raw output
        predicted_class = int(np.argmax(prediction))  # Find most probable class

        # Map prediction to sleep risk status
        risk_status = ["Healthy", "Mild Risk", "High Risk"][predicted_class]

        return {
            "prediction": predicted_class,
            "status": risk_status,
            "confidence": float(np.max(prediction))  # Confidence of prediction
        }
    
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
