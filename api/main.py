from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import json
import logging
import os
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import start_http_server
import time
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI app
app = FastAPI(title="IPL Win Predictor API", version="1.0.0")

# Prometheus metrics
PREDICTION_COUNTER = Counter('ipl_predictions_total', 'Total number of predictions made')
PREDICTION_LATENCY = Histogram('ipl_prediction_latency_seconds', 'Prediction latency in seconds')
MODEL_LOAD_TIME = Histogram('ipl_model_load_time_seconds', 'Model loading time in seconds')

# Global variables for model and scaler
model = None
scaler = None
model_info = None
label_encoders = None

class PredictionRequest(BaseModel):
    team1: str
    team2: str
    venue: str
    city: str
    team1_win_percentage: float
    team2_win_percentage: float
    team1_recent_form: Optional[float] = None
    team2_recent_form: Optional[float] = None
    team1_head_to_head: Optional[float] = None
    team2_head_to_head: Optional[float] = None

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    confidence: str
    features_used: List[str]

def load_model():
    """Load the trained model and artifacts."""
    global model, scaler, model_info, label_encoders

    start_time = time.time()

    try:
        # Load model
        with open('models/ipl_win_predictor.pkl', 'rb') as f:
            model = pickle.load(f)

        # Load scaler
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # Load model info
        with open('models/model_info.json', 'r') as f:
            model_info = json.load(f)

        # Load label encoders
        with open('data/features/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)

        load_time = time.time() - start_time
        MODEL_LOAD_TIME.observe(load_time)

        logging.info(f"Model loaded successfully in {load_time:.2f} seconds")

    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def prepare_features(request: PredictionRequest) -> np.ndarray:
    """Prepare features for prediction."""
    # Create feature dictionary
    features = {
        'team1_win_percentage': request.team1_win_percentage,
        'team2_win_percentage': request.team2_win_percentage,
        'team1_recent_form': request.team1_recent_form or 0.5,
        'team2_recent_form': request.team2_recent_form or 0.5,
        'team1_head_to_head': request.team1_head_to_head or 0.5,
        'team2_head_to_head': request.team2_head_to_head or 0.5,
    }

    # Encode categorical features
    if 'Team1_encoded' in model_info['feature_columns']:
        features['Team1_encoded'] = label_encoders['Team1'].transform([request.team1])[0]
    if 'Team2_encoded' in model_info['feature_columns']:
        features['Team2_encoded'] = label_encoders['Team2'].transform([request.team2])[0]
    if 'Venue_encoded' in model_info['feature_columns']:
        features['Venue_encoded'] = label_encoders['Venue'].transform([request.venue])[0]
    if 'City_encoded' in model_info['feature_columns']:
        features['City_encoded'] = label_encoders['City'].transform([request.city])[0]

    # Create interaction features
    features['win_percentage_diff'] = request.team1_win_percentage - request.team2_win_percentage
    features['recent_form_diff'] = (request.team1_recent_form or 0.5) - (request.team2_recent_form or 0.5)
    features['h2h_advantage'] = (request.team1_head_to_head or 0.5) - (request.team2_head_to_head or 0.5)

    # Create feature array in the correct order
    feature_array = []
    for feature in model_info['feature_columns']:
        if feature in features:
            feature_array.append(features[feature])
        else:
            feature_array.append(0)  # Default value for missing features

    return np.array(feature_array).reshape(1, -1)

def get_confidence_level(probability: float) -> str:
    """Get confidence level based on probability."""
    if probability >= 0.8 or probability <= 0.2:
        return "High"
    elif probability >= 0.6 or probability <= 0.4:
        return "Medium"
    else:
        return "Low"

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()
    # Start Prometheus metrics server
    start_http_server(8000)

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "IPL Win Predictor API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        # Prepare features
        features = prepare_features(request)

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        # Calculate latency
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        PREDICTION_COUNTER.inc()

        # Get confidence level
        confidence = get_confidence_level(probability)

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            confidence=confidence,
            features_used=model_info['feature_columns']
        )

    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get model information."""
    if model_info is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_name": model_info['model_name'],
        "model_type": model_info['model_type'],
        "training_date": model_info['training_date'],
        "feature_count": len(model_info['feature_columns']),
        "features": model_info['feature_columns']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)