import os
import numpy as np
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

# Define request body model
class HealthInput(BaseModel):
    heart_rate: float
    sleep_duration: float
    sleep_category: int  # Unused but included

@app.get("/")
def home():
    return {"message": "🏥 AI Health Tracker API is running!"}

@app.post("/predict")
def predict(data: HealthInput):
    try:
        # Normalize inputs
        input_data = scaler.transform([[data.heart_rate, data.sleep_duration]])
        
        # Reshape for LSTM
        input_data = np.reshape(input_data, (1, 1, 2))  # (batch_size, time_steps, features)
        
        # Get prediction
        prediction = model.predict(input_data)[0][0]
        risk_status = "At Risk" if prediction > 0.5 else "Healthy"
        
        return {"prediction": float(prediction), "status": risk_status}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)