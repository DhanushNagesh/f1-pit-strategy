from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import sys
from pathlib import Path

# Add src to path so we can import features
sys.path.append(str(Path(__file__).parent.parent))
from src.features import get_feature_columns

app = FastAPI(
    title="F1 Pit Stop Strategy API",
    description="Predict optimal pit stop windows using Bi-LSTM model",
    version="1.0.0"
)

# Load models at startup (paths relative to project root)
model_dir = Path(__file__).parent.parent / 'models'
baseline_model = joblib.load(model_dir / 'baseline_model.pkl')


class LapFeatures(BaseModel):
    """Input features for a single lap prediction."""
    tyre_age: int = Field(..., ge=0, le=50, description="Current tire age in laps")
    compound: str = Field(..., description="Tire compound (SOFT, MEDIUM, HARD)")
    lap_number_norm: float = Field(..., ge=0.0, le=1.0, description="Race completion (0-1)")
    position: int = Field(..., ge=1, le=20, description="Current race position")
    laps_in_current_stint: int = Field(..., ge=1, description="Laps completed on current tires")
    laps_past_expected_pit: float = Field(..., description="Laps past expected pit window")
    
    class Config:
        json_schema_extra = {
            "example": {
                "tyre_age": 18,
                "compound": "MEDIUM",
                "lap_number_norm": 0.35,
                "position": 5,
                "laps_in_current_stint": 18,
                "laps_past_expected_pit": -7
            }
        }


class PredictionResponse(BaseModel):
    """Prediction output."""
    pit_probability: float = Field(..., description="Probability of pitting this lap (0-1)")
    recommendation: str = Field(..., description="Human-readable recommendation")
    model_used: str = Field(..., description="Model that generated prediction")


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "F1 Pit Stop Strategy API",
        "models_loaded": ["baseline_rf"]
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_pit_stop(lap: LapFeatures):
    """
    Predict whether the driver should pit on this lap.
    
    Returns probability and recommendation based on current lap telemetry.
    """
    try:
        # Encode compound as one-hot
        compound_soft = 1 if lap.compound == "SOFT" else 0
        compound_medium = 1 if lap.compound == "MEDIUM" else 0
        compound_hard = 1 if lap.compound == "HARD" else 0
        
        # Build feature vector matching training order
        features = np.array([[
            lap.tyre_age,
            compound_soft,
            compound_medium,
            compound_hard,
            lap.lap_number_norm,
            lap.position,
            lap.laps_in_current_stint,
            lap.laps_past_expected_pit,
            0.0  # tire_deg_last_3 - set to 0 since API doesn't have historical laps
        ]])
        
        # Predict
        probability = float(baseline_model.predict_proba(features)[0][1])
        
        # Generate recommendation
        if probability > 0.5:
            recommendation = "PIT NOW - High probability of optimal pit window"
        elif probability > 0.3:
            recommendation = "CONSIDER PITTING - Approaching optimal window"
        else:
            recommendation = "STAY OUT - Not yet optimal for pit stop"
        
        return PredictionResponse(
            pit_probability=round(probability, 3),
            recommendation=recommendation,
            model_used="RandomForest"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/models/info")
def model_info():
    """Return information about loaded models."""
    import json
    
    metrics_path = Path(__file__).parent.parent / 'models' / 'baseline_metrics.json'
    with open(metrics_path, 'r') as f:
        baseline_metrics = json.load(f)
    
    return {
        "baseline": baseline_metrics,
        "features": get_feature_columns()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)