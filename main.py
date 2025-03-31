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

# Define request model (matching feature names used in training)
class HealthInput(BaseModel):
    Heart_Rate: float
    Sleep_Duration: float
    Timestamp_Numeric: int
    Sleep_Category: int = 0  # Default value

@app.get("/")
def home():
    return {"message": "🏥 AI Health Tracker API is running!"}

@app.post("/predict")
def predict(data: HealthInput):
    try:
        # Create DataFrame with correctly capitalized feature names
        input_df = pd.DataFrame([[data.Heart_Rate, data.Sleep_Duration, data.Timestamp_Numeric]],
                                columns=["Heart Rate", "Sleep Duration", "Timestamp_Numeric"])
        
        # Normalize inputs
        input_data = scaler.transform(input_df)

        # Reshape for LSTM (batch_size=1, time_steps=1, features=3)
        input_data = np.reshape(input_data, (1, 1, 3))

        # Get prediction
        prediction = model.predict(input_data)[0][0]
        risk_status = "At Risk" if prediction > 0.5 else "Healthy"

        # Debug logs (for testing)
        print(f"Received Data: {data}")
        print(f"Processed DataFrame:\n{input_df}")
        print(f"Scaled Input: {input_data}")
        print(f"Prediction: {prediction} -> Status: {risk_status}")

        return {"prediction": float(prediction), "status": risk_status}
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")  # Debugging log
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
