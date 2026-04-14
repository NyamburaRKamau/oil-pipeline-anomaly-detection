"""
Oil Pipeline Anomaly Detection - FastAPI Serving
Real-time inference endpoint for pipeline sensor data
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
import time
import logging
from datetime import datetime
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Oil Pipeline Anomaly Detection API",
    description="Real-time anomaly detection for oil pipeline sensor data",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model artifacts
MODEL_PATH = os.getenv("MODEL_PATH", "models/anomaly_detector.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load("models/feature_cols.pkl")
    logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.warning("Model not found. Run train.py first.")
    model = None
    scaler = None
    feature_cols = ['pressure_psi', 'temperature_c', 'flow_rate_m3hr', 'vibration_mms']


class SensorReading(BaseModel):
    pressure_psi: float = Field(..., description="Pipeline pressure in PSI", example=150.5)
    temperature_c: float = Field(..., description="Temperature in Celsius", example=60.2)
    flow_rate_m3hr: float = Field(..., description="Flow rate in m³/hr", example=200.1)
    vibration_mms: float = Field(..., description="Vibration in mm/s", example=0.5)
    pipeline_id: Optional[str] = Field(None, description="Pipeline identifier", example="PIPE-001")


class BatchSensorReadings(BaseModel):
    readings: list[SensorReading]


class AnomalyPrediction(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    severity: str
    timestamp: str
    pipeline_id: Optional[str]
    sensor_values: dict
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    timestamp: str


def get_severity(anomaly_score: float) -> str:
    """Classify anomaly severity based on score"""
    if anomaly_score < 0.3:
        return "LOW"
    elif anomaly_score < 0.6:
        return "MEDIUM"
    else:
        return "HIGH"


def predict_anomaly(reading: SensorReading):
    """Run inference on a single sensor reading"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    features = np.array([[
        reading.pressure_psi,
        reading.temperature_c,
        reading.flow_rate_m3hr,
        reading.vibration_mms
    ]])

    features_scaled = scaler.transform(features)

    # Isolation Forest: -1 = anomaly, 1 = normal
    prediction = model.predict(features_scaled)[0]
    raw_score = -model.score_samples(features_scaled)[0]

    is_anomaly = prediction == -1
    normalized_score = min(max(raw_score / 0.8, 0), 1)  # Normalize to 0-1

    return is_anomaly, normalized_score


@app.get("/", response_model=HealthResponse)
def root():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/predict", response_model=AnomalyPrediction)
def predict(reading: SensorReading):
    """
    Predict anomaly for a single pipeline sensor reading.

    Returns anomaly status, severity, and score.
    """
    start_time = time.time()

    is_anomaly, score = predict_anomaly(reading)
    severity = get_severity(score) if is_anomaly else "NONE"

    if is_anomaly:
        message = f"⚠️ ANOMALY DETECTED - Severity: {severity}. Immediate inspection recommended."
    else:
        message = "✅ Normal operation. All sensors within acceptable range."

    latency = (time.time() - start_time) * 1000
    logger.info(f"Prediction: anomaly={is_anomaly}, score={score:.3f}, latency={latency:.1f}ms, pipeline={reading.pipeline_id}")

    return AnomalyPrediction(
        is_anomaly=is_anomaly,
        anomaly_score=round(score, 4),
        severity=severity,
        timestamp=datetime.utcnow().isoformat(),
        pipeline_id=reading.pipeline_id,
        sensor_values={
            "pressure_psi": reading.pressure_psi,
            "temperature_c": reading.temperature_c,
            "flow_rate_m3hr": reading.flow_rate_m3hr,
            "vibration_mms": reading.vibration_mms
        },
        message=message
    )


@app.post("/predict/batch")
def predict_batch(batch: BatchSensorReadings):
    """
    Predict anomalies for a batch of sensor readings.
    """
    results = []
    anomaly_count = 0

    for reading in batch.readings:
        is_anomaly, score = predict_anomaly(reading)
        severity = get_severity(score) if is_anomaly else "NONE"

        if is_anomaly:
            anomaly_count += 1

        results.append({
            "pipeline_id": reading.pipeline_id,
            "is_anomaly": is_anomaly,
            "anomaly_score": round(score, 4),
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat()
        })

    return {
        "total_readings": len(batch.readings),
        "anomalies_detected": anomaly_count,
        "anomaly_rate": round(anomaly_count / len(batch.readings), 3),
        "results": results
    }


@app.get("/model/info")
def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": type(model).__name__,
        "features": feature_cols,
        "contamination": model.contamination,
        "n_estimators": model.n_estimators,
        "status": "loaded"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
