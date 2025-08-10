import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from tensorflow.keras.models import load_model
import joblib

# --------------------
# Simulation Settings
# --------------------
def generate_packet():
    """Simulate a packet with random features."""
    is_botnet = np.random.choice([0, 1], p=[0.7, 0.3])  # 70% normal, 30% botnet
    if is_botnet == 0:
        packet_size = np.random.normal(500, 150)
        interval = np.random.exponential(2.0)
    else:
        packet_size = np.random.normal(800, 100)
        interval = np.random.exponential(0.5)
    return abs(packet_size), abs(interval), is_botnet

# --------------------
# Load Model and Scaler
# --------------------
@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists("botnet_detection_model.h5") or not os.path.exists("scaler.pkl"):
        st.error("Model and scaler not found. Please run `main.py` first to train and save them.")
        st.stop()
    model = load_model("botnet_detection_model.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="Botnet Detection Simulation", layout="wide")
st.title("🤖 Real-Time Botnet Detection Simulation")

st.sidebar.header("⚙️ Simulation Settings")
ticks = st.sidebar.number_input("Number of packets", min_value=5, max_value=200, value=20)
delay = st.sidebar.slider("Delay between packets (seconds)", min_value=0.1, max_value=3.0, value=1.0)
run_button = st.sidebar.button("Start Simulation")

placeholder_table = st.empty()
placeholder_chart = st.empty()

# DataFrame to store results
df_sim = pd.DataFrame(columns=["Packet Size", "Interval", "Prediction", "Confidence", "Actual"])

if run_button:
    progress_bar = st.progress(0)
    for i in range(ticks):
        packet_size, interval, actual_label = generate_packet()
        X_new = scaler.transform([[packet_size, interval]])
        pred_prob = model.predict(X_new, verbose=0)[0][0]
        pred_class = int(pred_prob > 0.5)
        confidence = pred_prob if pred_class == 1 else 1 - pred_prob

        status = "BOTNET ⚠️" if pred_class == 1 else "NORMAL ✅"

        # Append to DataFrame
        df_sim.loc[len(df_sim)] = [
            round(packet_size, 2),
            round(interval, 2),
            status,
            round(confidence * 100, 2),
            "BOTNET" if actual_label == 1 else "NORMAL"
        ]

        # Update UI table
        placeholder_table.dataframe(df_sim.tail(10))

        # Update chart
        chart_data = df_sim["Prediction"].value_counts().reset_index()
        chart_data.columns = ["Prediction", "Count"]
        placeholder_chart.bar_chart(chart_data.set_index("Prediction"))

        progress_bar.progress((i + 1) / ticks)
        time.sleep(delay)

    st.success("✅ Simulation complete!")

# --------------------
# Show Final Stats
# --------------------
if not df_sim.empty:
    st.subheader("📊 Final Simulation Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Packets", len(df_sim))
    col2.metric("Detected Botnets", (df_sim["Prediction"] == "BOTNET ⚠️").sum())
    col3.metric("Detected Normal", (df_sim["Prediction"] == "NORMAL ✅").sum())

    st.subheader("🔍 Detailed Results")
    st.dataframe(df_sim)
