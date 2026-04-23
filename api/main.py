from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import joblib
import numpy as np
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.features import get_feature_columns

import torch.nn as nn

class PitStopLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)
    
app = FastAPI(
    title="F1 Pit Stop Strategy API",
    description="Predict optimal pit stop windows using Bi-LSTM sequential model",
    version="2.0.0"
)

# Load models at startup
model_dir = Path(__file__).parent.parent / 'models'

# Load LSTM model
device = torch.device('cpu')
input_size = len(get_feature_columns())
lstm_model = PitStopLSTM(input_size=input_size, hidden_size=64, num_layers=2)
lstm_model.load_state_dict(torch.load(model_dir / 'lstm_model.pth', map_location=device))
lstm_model.eval()

# Load scaler
scaler = joblib.load(model_dir / 'scaler.pkl')


class LapData(BaseModel):
    """Single lap telemetry."""
    lap_number: int = Field(..., ge=1, description="Lap number in race")
    tyre_age: int = Field(..., ge=1, le=50, description="Tire age in laps")
    compound: str = Field(..., description="Tire compound (SOFT, MEDIUM, HARD)")
    position: int = Field(..., ge=1, le=20, description="Race position")
    
    class Config:
        json_schema_extra = {
            "example": {
                "lap_number": 18,
                "tyre_age": 18,
                "compound": "MEDIUM",
                "position": 5
            }
        }


class StintSequence(BaseModel):
    """Sequence of laps in current stint."""
    laps: List[LapData] = Field(..., min_items=3, max_items=30, description="Last 3-30 laps of current stint")
    total_race_laps: int = Field(..., ge=50, le=80, description="Total laps in race")
    
    class Config:
        json_schema_extra = {
            "example": {
                "laps": [
                    {"lap_number": 16, "tyre_age": 16, "compound": "MEDIUM", "position": 5},
                    {"lap_number": 17, "tyre_age": 17, "compound": "MEDIUM", "position": 5},
                    {"lap_number": 18, "tyre_age": 18, "compound": "MEDIUM", "position": 5}
                ],
                "total_race_laps": 57
            }
        }


class PredictionResponse(BaseModel):
    """Prediction output."""
    pit_probability: float = Field(..., description="Probability of pitting next lap (0-1)")
    recommendation: str = Field(..., description="Human-readable recommendation")
    model_used: str = Field(..., description="Model that generated prediction")
    sequence_length: int = Field(..., description="Number of laps analyzed")


def encode_lap_features(lap: LapData, total_race_laps: int) -> np.ndarray:
    """Convert lap data to feature vector."""
    compound_soft = 1 if lap.compound == "SOFT" else 0
    compound_medium = 1 if lap.compound == "MEDIUM" else 0
    compound_hard = 1 if lap.compound == "HARD" else 0
    
    lap_number_norm = lap.lap_number / total_race_laps
    
    # Expected pit windows
    expected_pit_window = {"SOFT": 15, "MEDIUM": 25, "HARD": 35}.get(lap.compound, 25)
    laps_past_expected_pit = lap.tyre_age - expected_pit_window
    
    laps_in_current_stint = lap.tyre_age  # Assuming tire age = stint length
    
    # Note: tire_deg_last_3 set to 0 since we don't have lap times in API
    tire_deg = 0.0
    
    return np.array([
        lap.tyre_age,
        compound_soft,
        compound_medium,
        compound_hard,
        lap_number_norm,
        lap.position,
        laps_in_current_stint,
        laps_past_expected_pit,
        tire_deg
    ])


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "F1 Pit Stop Strategy API",
        "model": "Bi-LSTM (F1=0.93)",
        "input_format": "sequence_of_laps"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_pit_stop(stint: StintSequence):
    """
    Predict whether the driver should pit after the current stint.
    
    Accepts a sequence of laps (minimum 3, maximum 30) representing the current stint.
    The model analyzes the entire sequence to predict if a pit stop is optimal.
    """
    try:
        # Convert laps to feature matrix
        sequence = []
        for lap in stint.laps:
            features = encode_lap_features(lap, stint.total_race_laps)
            sequence.append(features)
        
        sequence = np.array(sequence)  # Shape: (seq_len, features)
        
        # Scale features
        sequence_scaled = scaler.transform(sequence)
        
        # Pad to max length (30 laps)
        max_len = 30
        padded = np.zeros((max_len, sequence_scaled.shape[1]))
        seq_len = min(len(sequence_scaled), max_len)
        padded[:seq_len] = sequence_scaled[:seq_len]
        
        # Convert to tensor and add batch dimension
        X = torch.FloatTensor(padded).unsqueeze(0)  # Shape: (1, 30, features)
        
        # Predict
        with torch.no_grad():
            probability = float(lstm_model(X).squeeze().item())
        
        # Generate recommendation
        if probability > 0.7:
            recommendation = "PIT NOW - High probability of optimal pit window"
        elif probability > 0.5:
            recommendation = "CONSIDER PITTING - Entering optimal window"
        elif probability > 0.3:
            recommendation = "MONITOR - Approaching pit window"
        else:
            recommendation = "STAY OUT - Not yet optimal for pit stop"
        
        return PredictionResponse(
            pit_probability=round(probability, 3),
            recommendation=recommendation,
            model_used="Bi-LSTM",
            sequence_length=len(stint.laps)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/models/info")
def model_info():
    """Return information about the LSTM model."""
    import json
    
    metrics_path = Path(__file__).parent.parent / 'models' / 'lstm_metrics.json'
    with open(metrics_path, 'r') as f:
        lstm_metrics = json.load(f)
    
    return {
        "model": "Bi-LSTM",
        "metrics": lstm_metrics,
        "features": get_feature_columns(),
        "input_requirements": {
            "min_laps": 3,
            "max_laps": 30,
            "description": "Provide sequence of recent laps in current stint"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)