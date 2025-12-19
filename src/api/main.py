"""FastAPI application for microplastic prediction API."""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from loguru import logger
import joblib
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

from ..utils.config import load_config, get_data_paths
from ..models.spectra_models import Spectra1DCNN
from ..models.image_models import EfficientNetClassifier, MicroplasticDetector
from ..models.tabular_models import TabularXGBoost

# Initialize FastAPI app
app = FastAPI(title="Microplastic Detection API", version="0.1.0")

# Load configuration
config = load_config()
paths = get_data_paths(config)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("api", {}).get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
prediction_counter = Counter(
    'microplastic_predictions_total',
    'Total number of predictions',
    ['endpoint', 'status']
)
prediction_latency = Histogram(
    'microplastic_prediction_latency_seconds',
    'Prediction latency',
    ['endpoint']
)

# Load models (lazy loading)
models = {}


def load_models():
    """Load trained models."""
    global models
    
    # Load tabular model
    tabular_model_path = paths["models"] / "tabular_xgboost.pkl"
    if tabular_model_path.exists():
        models["tabular"] = joblib.load(tabular_model_path)
        logger.info("Loaded tabular model")
    
    # Load spectra model (if available)
    spectra_model_path = paths["models"] / "spectra_cnn.pt"
    if spectra_model_path.exists():
        models["spectra"] = torch.load(spectra_model_path, map_location="cpu")
        models["spectra"].eval()
        logger.info("Loaded spectra model")
    
    # Load image model (if available)
    image_model_path = paths["models"] / "image_classifier.pt"
    if image_model_path.exists():
        models["image"] = torch.load(image_model_path, map_location="cpu")
        models["image"].eval()
        logger.info("Loaded image model")
    
    # Load YOLOv8 detector (if available)
    detector_path = paths["models"] / "yolov8_detector.pt"
    if detector_path.exists():
        from ..models.image_models import MicroplasticDetector
        models["detector"] = MicroplasticDetector(model_size="n", pretrained=False)
        models["detector"].model = torch.load(detector_path, map_location="cpu")
        logger.info("Loaded YOLOv8 detector")


# Request/Response models
class TabularInput(BaseModel):
    """Tabular data input."""
    features: Dict[str, float]


class SpectraInput(BaseModel):
    """Spectral data input."""
    wavelength: List[float]
    intensity: List[float]


class PredictionResponse(BaseModel):
    """Prediction response."""
    prediction: Any
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    total: int


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    load_models()
    logger.info("API started")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Microplastic Detection API",
        "version": "0.1.0",
        "endpoints": {
            "predict_presence": "/api/v1/predict/presence",
            "predict_type": "/api/v1/predict/type",
            "predict_concentration": "/api/v1/predict/concentration",
            "batch_predict": "/api/v1/predict/batch",
            "health": "/health",
            "metrics": "/metrics"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys())
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")


@app.post("/api/v1/predict/presence", response_model=PredictionResponse)
async def predict_presence(input_data: TabularInput):
    """Predict presence/absence of microplastics."""
    import time
    start_time = time.time()
    
    try:
        if "tabular" not in models:
            raise HTTPException(status_code=503, detail="Tabular model not loaded")
        
        # Convert input to DataFrame
        df = pd.DataFrame([input_data.features])
        
        # Make prediction
        model = models["tabular"]
        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0] if hasattr(model, 'predict_proba') else None
        
        # Calculate confidence
        confidence = float(np.max(proba)) if proba is not None else None
        
        # Format probabilities
        probabilities = None
        if proba is not None:
            probabilities = {
                "absent": float(proba[0]),
                "present": float(proba[1])
            }
        
        latency = time.time() - start_time
        prediction_latency.labels(endpoint="presence").observe(latency)
        prediction_counter.labels(endpoint="presence", status="success").inc()
        
        return PredictionResponse(
            prediction=int(prediction),
            confidence=confidence,
            probabilities=probabilities
        )
    
    except Exception as e:
        prediction_counter.labels(endpoint="presence", status="error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predict/concentration", response_model=PredictionResponse)
async def predict_concentration(input_data: TabularInput):
    """Predict microplastic concentration."""
    import time
    start_time = time.time()
    
    try:
        if "tabular" not in models:
            raise HTTPException(status_code=503, detail="Tabular model not loaded")
        
        # Convert input to DataFrame
        df = pd.DataFrame([input_data.features])
        
        # Make prediction
        model = models["tabular"]
        prediction = model.predict(df)[0]
        
        latency = time.time() - start_time
        prediction_latency.labels(endpoint="concentration").observe(latency)
        prediction_counter.labels(endpoint="concentration", status="success").inc()
        
        return PredictionResponse(
            prediction=float(prediction),
            confidence=None
        )
    
    except Exception as e:
        prediction_counter.labels(endpoint="concentration", status="error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predict/type", response_model=PredictionResponse)
async def predict_type(spectra: SpectraInput):
    """Predict microplastic polymer type from spectrum."""
    import time
    start_time = time.time()
    
    try:
        if "spectra" not in models:
            raise HTTPException(status_code=503, detail="Spectra model not loaded")
        
        # Preprocess spectrum
        wavelength = np.array(spectra.wavelength)
        intensity = np.array(spectra.intensity)
        
        # Resample if needed (simplified)
        # In practice, use the same preprocessing as training
        
        # Convert to tensor
        intensity_tensor = torch.FloatTensor(intensity).unsqueeze(0).unsqueeze(0)
        
        # Make prediction
        model = models["spectra"]
        with torch.no_grad():
            output = model(intensity_tensor)
            probabilities = torch.exp(output).numpy()[0]
            prediction = int(np.argmax(probabilities))
        
        # Map to polymer types (example)
        polymer_types = ["PE", "PP", "PS", "PET", "PVC", "Other"]
        predicted_type = polymer_types[prediction] if prediction < len(polymer_types) else "Unknown"
        
        latency = time.time() - start_time
        prediction_latency.labels(endpoint="type").observe(latency)
        prediction_counter.labels(endpoint="type", status="success").inc()
        
        return PredictionResponse(
            prediction=predicted_type,
            confidence=float(np.max(probabilities)),
            probabilities={polymer_types[i]: float(probabilities[i]) 
                          for i in range(min(len(polymer_types), len(probabilities)))}
        )
    
    except Exception as e:
        prediction_counter.labels(endpoint="type", status="error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """Detect microplastics in an uploaded image."""
    import time
    from PIL import Image
    import io
    
    start_time = time.time()
    
    try:
        if "detector" not in models:
            raise HTTPException(status_code=503, detail="Image detector not loaded")
        
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save temporarily
        temp_path = Path("/tmp") / file.filename
        image.save(temp_path)
        
        # Detect
        detector = models["detector"]
        detections = detector.predict(str(temp_path), conf_threshold=0.5)
        
        # Clean up
        temp_path.unlink()
        
        latency = time.time() - start_time
        prediction_latency.labels(endpoint="image").observe(latency)
        prediction_counter.labels(endpoint="image", status="success").inc()
        
        return PredictionResponse(
            prediction={
                "count": int(detections["count"]),
                "boxes": detections["boxes"].tolist() if len(detections["boxes"]) > 0 else [],
                "scores": detections["scores"].tolist() if len(detections["scores"]) > 0 else []
            },
            confidence=float(np.mean(detections["scores"])) if len(detections["scores"]) > 0 else 0.0
        )
    
    except Exception as e:
        prediction_counter.labels(endpoint="image", status="error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(inputs: List[TabularInput]):
    """Batch prediction endpoint."""
    import time
    start_time = time.time()
    
    try:
        if "tabular" not in models:
            raise HTTPException(status_code=503, detail="Tabular model not loaded")
        
        # Convert to DataFrame
        df = pd.DataFrame([inp.features for inp in inputs])
        
        # Make predictions
        model = models["tabular"]
        predictions = model.predict(df)
        probas = model.predict_proba(df) if hasattr(model, 'predict_proba') else None
        
        # Format responses
        results = []
        for i, pred in enumerate(predictions):
            proba = probas[i] if probas is not None else None
            confidence = float(np.max(proba)) if proba is not None else None
            probabilities = {
                "absent": float(proba[0]),
                "present": float(proba[1])
            } if proba is not None else None
            
            results.append(PredictionResponse(
                prediction=int(pred),
                confidence=confidence,
                probabilities=probabilities
            ))
        
        latency = time.time() - start_time
        prediction_latency.labels(endpoint="batch").observe(latency)
        prediction_counter.labels(endpoint="batch", status="success").inc()
        
        return BatchPredictionResponse(
            predictions=results,
            total=len(results)
        )
    
    except Exception as e:
        prediction_counter.labels(endpoint="batch", status="error").inc()
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    api_config = config.get("api", {})
    uvicorn.run(
        "src.api.main:app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        reload=True
    )

