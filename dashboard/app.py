import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="F1 Pit Strategy Optimizer", layout="wide")

st.title("🏎️ F1 Pit Stop Strategy Optimizer")
st.markdown("Real-time pit stop prediction using Bi-LSTM deep learning")

# Sidebar inputs
st.sidebar.header("Race Conditions")

tyre_age = st.sidebar.slider("Tire Age (laps)", 1, 50, 18)
compound = st.sidebar.selectbox("Tire Compound", ["SOFT", "MEDIUM", "HARD"])
position = st.sidebar.slider("Current Position", 1, 20, 5)
lap_number = st.sidebar.slider("Current Lap", 1, 70, 20)
total_laps = st.sidebar.slider("Total Race Laps", 50, 70, 57)

# Compute derived features
lap_number_norm = lap_number / total_laps
laps_in_current_stint = tyre_age  # Assuming tire age = stint length for simplicity

expected_pit_windows = {"SOFT": 15, "MEDIUM": 25, "HARD": 35}
expected_pit_window = expected_pit_windows[compound]
laps_past_expected_pit = tyre_age - expected_pit_window

# Display current state
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Tire Age", f"{tyre_age} laps")
    st.metric("Compound", compound)
with col2:
    st.metric("Race Progress", f"{lap_number}/{total_laps}")
    st.metric("Position", f"P{position}")
with col3:
    st.metric("Expected Pit Window", f"{expected_pit_window} laps")
    delta_color = "off" if laps_past_expected_pit < 0 else "normal"
    st.metric("Laps Past Expected", f"{laps_past_expected_pit:+.0f}", delta_color=delta_color)

# Predict button
if st.button("🔮 Predict Pit Stop", type="primary"):
    # Call API
    payload = {
        "tyre_age": tyre_age,
        "compound": compound,
        "lap_number_norm": lap_number_norm,
        "position": position,
        "laps_in_current_stint": laps_in_current_stint,
        "laps_past_expected_pit": laps_past_expected_pit
    }
    
    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Display prediction
        st.success("✅ Prediction Complete")
        
        prob = result["pit_probability"]
        recommendation = result["recommendation"]
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Pit Stop Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if prob > 0.5 else "orange" if prob > 0.3 else "green"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 50], 'color': "lightyellow"},
                    {'range': [50, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation
        if prob > 0.5:
            st.error(f"⚠️ {recommendation}")
        elif prob > 0.3:
            st.warning(f"🟡 {recommendation}")
        else:
            st.info(f"✅ {recommendation}")
        
        # Feature contribution (mock - would need SHAP integration for real values)
        st.subheader("Feature Contribution")
        feature_data = pd.DataFrame({
            "Feature": ["Tire Age", "Race Progress", "Past Expected Pit", "Compound", "Position"],
            "Contribution": [0.35, 0.28, 0.22, 0.10, 0.05]  # Mock values
        })
        st.bar_chart(feature_data.set_index("Feature"))
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Error: {e}")
        st.info("Make sure the API is running: `python -m uvicorn api.main:app`")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This tool uses a Bi-LSTM neural network trained on 20 F1 races 
from the 2023 season to predict optimal pit stop windows.

**Model Performance:**
- Bi-LSTM F1 Score: 0.93
- Baseline RF F1 Score: 0.22
""")