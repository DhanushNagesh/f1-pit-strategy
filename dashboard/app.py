import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="F1 Pit Strategy Optimizer", layout="wide")

st.title("🏎️ F1 Pit Stop Strategy Optimizer")
st.markdown("Real-time pit stop prediction using **Bi-LSTM** deep learning (F1-score: 0.93)")

# Sidebar inputs
st.sidebar.header("Current Race State")

current_lap = st.sidebar.slider("Current Lap", 1, 70, 18)
total_laps = st.sidebar.slider("Total Race Laps", 50, 70, 57)
position = st.sidebar.slider("Current Position", 1, 20, 5)

st.sidebar.markdown("---")
st.sidebar.header("Current Stint")

stint_start_lap = st.sidebar.slider("Stint Started at Lap", 1, current_lap, max(1, current_lap - 17))
compound = st.sidebar.selectbox("Tire Compound", ["SOFT", "MEDIUM", "HARD"])

# Calculate stint length
stint_length = current_lap - stint_start_lap + 1
tyre_age = stint_length

st.sidebar.metric("Stint Length", f"{stint_length} laps")
st.sidebar.metric("Tire Age", f"{tyre_age} laps")

# Expected pit window
expected_pit_windows = {"SOFT": 15, "MEDIUM": 25, "HARD": 35}
expected_pit_window = expected_pit_windows[compound]
laps_past_expected = tyre_age - expected_pit_window

st.sidebar.metric("Expected Pit Window", f"{expected_pit_window} laps")
st.sidebar.metric("Deviation from Expected", f"{laps_past_expected:+d} laps", 
                  delta_color="inverse" if laps_past_expected > 0 else "off")

# Display current state
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Race Progress", f"{current_lap}/{total_laps} laps")
    st.metric("Progress %", f"{(current_lap/total_laps)*100:.1f}%")
with col2:
    st.metric("Position", f"P{position}")
    st.metric("Compound", compound)
with col3:
    st.metric("Current Lap", current_lap)
    st.metric("Tire Age", f"{tyre_age} laps")

# Build sequence (last N laps of current stint)
sequence_length = min(stint_length, 10)  # Use last 10 laps or entire stint if shorter

st.info(f"📊 Analyzing last **{sequence_length} laps** of current stint for prediction")

# Generate lap sequence
laps_sequence = []
for i in range(sequence_length):
    lap_num = current_lap - (sequence_length - 1) + i
    lap_tyre_age = stint_start_lap - stint_start_lap + (lap_num - stint_start_lap + 1)
    
    laps_sequence.append({
        "lap_number": lap_num,
        "tyre_age": lap_tyre_age,
        "compound": compound,
        "position": position
    })

# Show sequence
with st.expander("🔍 View Stint Sequence Being Analyzed"):
    seq_df = pd.DataFrame(laps_sequence)
    st.dataframe(seq_df, use_container_width=True)

# Predict button
if st.button("🔮 Predict Pit Stop", type="primary", use_container_width=True):
    # Call API
    payload = {
        "laps": laps_sequence,
        "total_race_laps": total_laps
    }
    
    try:
        with st.spinner("Running Bi-LSTM inference..."):
            response = requests.post("http://localhost:8000/predict", json=payload)
            response.raise_for_status()
            result = response.json()
        
        # Display prediction
        st.success("✅ Prediction Complete")
        
        prob = result["pit_probability"]
        recommendation = result["recommendation"]
        model = result["model_used"]
        seq_len = result["sequence_length"]
        
        st.caption(f"Model: {model} | Analyzed {seq_len} laps")
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Pit Stop Probability", 'font': {'size': 20}},
            number={'suffix': "%", 'font': {'size': 40}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "darkred" if prob > 0.7 else "orange" if prob > 0.5 else "gold" if prob > 0.3 else "green"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 50], 'color': "lightyellow"},
                    {'range': [50, 70], 'color': "lightcoral"},
                    {'range': [70, 100], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation box
        if prob > 0.7:
            st.error(f"🔴 **{recommendation}**")
        elif prob > 0.5:
            st.warning(f"🟠 **{recommendation}**")
        elif prob > 0.3:
            st.info(f"🟡 **{recommendation}**")
        else:
            st.success(f"🟢 **{recommendation}**")
        
        # Additional context
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pit Probability", f"{prob*100:.1f}%")
            st.metric("Sequence Length", f"{seq_len} laps")
        with col2:
            expected_next_pit = expected_pit_window - tyre_age
            if expected_next_pit > 0:
                st.metric("Laps to Expected Pit", f"{expected_next_pit}")
            else:
                st.metric("Laps Overdue", f"{abs(expected_next_pit)}", delta_color="inverse")
        
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API")
        st.info("**Start the API first:**\n```bash\npython -m uvicorn api.main:app --reload\n```")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Error: {e}")
        if hasattr(e.response, 'text'):
            st.code(e.response.text)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.markdown("""
**Bi-LSTM (Sequential)**
- F1 Score: **0.93**
- Precision: **0.90**
- Recall: **0.96**

**Baseline RF (Lap-level)**
- F1 Score: 0.22
- Precision: 0.14
- Recall: 0.48

**Key Insight:** Sequential modeling captures temporal pit strategy patterns that traditional ML misses.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("""
This system uses a **Bi-LSTM neural network** trained on 20 F1 races (2023 season) to predict optimal pit windows.

The model processes **stint sequences** bidirectionally to capture tire degradation patterns over time.
""")