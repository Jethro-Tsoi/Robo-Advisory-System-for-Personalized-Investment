from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict
import json

app = FastAPI(title="Financial Sentiment Analysis API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelMetrics(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float

class PredictionResult(BaseModel):
    text: str
    gamma_prediction: str
    finbert_prediction: str
    gamma_confidence: float
    finbert_confidence: float

@app.get("/")
async def root():
    return {"message": "Financial Sentiment Analysis API"}

@app.get("/metrics", response_model=Dict[str, ModelMetrics])
async def get_model_metrics():
    """Get evaluation metrics for both models"""
    metrics = {}
    
    # Load metrics from saved CSV files
    models_dir = Path("../../models")
    
    for model in ['gamma3', 'finbert']:
        metrics_path = models_dir / model / "metrics.csv"
        try:
            df = pd.read_csv(metrics_path)
            metrics[model] = ModelMetrics(
                model_name=model,
                accuracy=float(df['accuracy'].iloc[0]),
                precision=float(df['precision'].iloc[0]),
                recall=float(df['recall'].iloc[0]),
                f1=float(df['f1'].iloc[0])
            )
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Metrics for {model} not found")
    
    return metrics

@app.get("/confusion_matrices")
async def get_confusion_matrices():
    """Get confusion matrices for both models"""
    matrices = {}
    models_dir = Path("../../models")
    
    for model in ['gamma3', 'finbert']:
        matrix_path = models_dir / model / "confusion_matrix.npy"
        try:
            cm = np.load(matrix_path)
            matrices[model] = cm.tolist()
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Confusion matrix for {model} not found")
    
    return matrices

@app.get("/sample_predictions", response_model=List[PredictionResult])
async def get_sample_predictions():
    """Get sample predictions from both models"""
    try:
        predictions_path = Path("../../data/sample_predictions.json")
        with open(predictions_path) as f:
            predictions = json.load(f)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=404, detail="Sample predictions not found")

@app.get("/performance_comparison")
async def get_performance_comparison():
    """Get detailed performance comparison between models"""
    try:
        comparison_path = Path("../../data/model_comparison.json")
        with open(comparison_path) as f:
            comparison = json.load(f)
        return comparison
    except Exception as e:
        raise HTTPException(status_code=404, detail="Performance comparison data not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
