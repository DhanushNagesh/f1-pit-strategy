# 🏎️ F1 Pit Stop Strategy Optimizer


---

**F1 Pit Stop Strategy Optimizer** - Machine learning system for predicting optimal Formula 1 pit stop windows using Bi-LSTM sequential deep learning on real race telemetry.

## Table of Contents
  * [Quick Start](#quick-start)
  * [Models](#models)
  * [Tech Stack](#tech-stack)
  * [Project Structure](#project-structure)
  * [Results](#results)
  * [API Usage](#api-usage)
  * [Example Usage](#example-usage)
  * [Future Work](#future-work)

## Quick Start

```bash
# Clone and install
git clone https://github.com/DhanushNagesh/f1-pit-strategy.git
cd f1-pit-strategy
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run API
python -m uvicorn api.main:app --reload

# Run Dashboard
streamlit run dashboard/app.py
```

## Models

### Deep Learning Models
1. **Bi-LSTM** (Primary Model)
   - F1-Score: **0.93**
   - Precision: 0.90
   - Recall: 0.96
   - Architecture: 2-layer bidirectional LSTM, 64 hidden units

2. **Random Forest** (Baseline)
   - F1-Score: 0.22
   - Precision: 0.14
   - Recall: 0.48

3. **XGBoost** (Baseline)
   - F1-Score: 0.22
   - Precision: 0.14
   - Recall: 0.58

**Key Finding**: Baseline models fail (F1=0.22) because pit stops require **sequential context**. The Bi-LSTM's 4x improvement validates temporal modeling.

## Tech Stack

**Data & ML**
- Python 3.9+
- PyTorch 2.0
- FastF1 (F1 telemetry API)
- scikit-learn, XGBoost
- SHAP (explainability)

**Deployment**
- FastAPI (REST API)
- Streamlit (dashboard)
- Plotly (visualization)

## Project Structure
f1-pit-strategy/
├── src/
│   ├── data_pipeline.py       # FastF1 data ingestion
│   ├── features.py             # Feature engineering
│   ├── train.py                # Baseline training
│   ├── train_lstm.py           # Bi-LSTM training
│   └── explainability.py       # SHAP analysis
├── api/
│   └── main.py                 # FastAPI endpoint
├── dashboard/
│   └── app.py                  # Streamlit UI
├── models/                     # Saved weights
└── data/                       # Race data

## Results

### Model Comparison

| Model | F1 | Precision | Recall | Approach |
|-------|-----|-----------|--------|----------|
| Random Forest | 0.22 | 0.14 | 0.48 | Lap-level features |
| XGBoost | 0.22 | 0.14 | 0.58 | Lap-level features |
| **Bi-LSTM** | **0.93** | **0.90** | **0.96** | Sequential stints |

### Feature Importance (SHAP)

Top predictive features:
1. `lap_number_norm` (0.113) - Strategic pit windows
2. `tire_deg_last_3` (0.063) - Pace degradation
3. `laps_past_expected_pit` (0.053) - Overdue signal
4. `tyre_age` (0.046) - Tire wear

### Dataset
- **Source**: FastF1 API
- **Coverage**: 20 races (2023 season)
- **Size**: 22,319 laps, 874 pit stops
- **Class balance**: 3.9% positive (pit laps)

## API Usage

### Request Format

The API accepts a **sequence of laps** (3-30 laps):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "laps": [
      {"lap_number": 16, "tyre_age": 16, "compound": "MEDIUM", "position": 5},
      {"lap_number": 17, "tyre_age": 17, "compound": "MEDIUM", "position": 5},
      {"lap_number": 18, "tyre_age": 18, "compound": "MEDIUM", "position": 5}
    ],
    "total_race_laps": 57
  }'
```

### Response

```json
{
  "pit_probability": 0.943,
  "recommendation": "PIT NOW - High probability of optimal pit window",
  "model_used": "Bi-LSTM",
  "sequence_length": 3
}
```

### Interactive Docs

Visit `http://localhost:8000/docs` 

## Example Usage
<img width="2492" height="1236" alt="image" src="https://github.com/user-attachments/assets/01b25be5-fb57-431a-96de-267b46bbc7de" />

<img width="2482" height="1258" alt="image" src="https://github.com/user-attachments/assets/08e68d88-b78e-4478-9b05-2028d7e2609e" />

## Future Work

1. **Competitor Context**
   - Gap to car ahead/behind
   - Opponent pit status
   - Undercut/overcut modeling

2. **Multi-Circuit Analysis**
   - Test generalization across track types
   - Street circuits vs high-speed tracks

3. **Strategy Simulation**
   - Compare multiple pit window scenarios
   - Expected finishing position outputs

4. **Deployment**
   - Docker containerization
   - AWS/Hugging Face Spaces hosting

## Contact

**Dhanush Nagesh** - [@DhanushNagesh](https://github.com/DhanushNagesh)

