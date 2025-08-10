import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --------------------
# Fallback Data Generator
# --------------------
def create_sample_data(n_samples=1000):
    np.random.seed(42)
    normal_packet_sizes = np.random.normal(500, 150, n_samples // 2)
    normal_intervals = np.random.exponential(2.0, n_samples // 2)
    botnet_packet_sizes = np.random.normal(800, 100, n_samples // 2)
    botnet_intervals = np.random.exponential(0.5, n_samples // 2)

    packet_sizes = np.abs(np.concatenate([normal_packet_sizes, botnet_packet_sizes]))
    intervals = np.abs(np.concatenate([normal_intervals, botnet_intervals]))
    labels = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

    return pd.DataFrame({
        'packet_size': packet_sizes,
        'interval': intervals,
        'is_botnet': labels.astype(int)
    }).sample(frac=1).reset_index(drop=True)

# --------------------
# Load Dataset with Guaranteed Output
# --------------------
@st.cache_data
def load_data():
    if os.path.exists("synthetic_robot_logs.csv"):
        try:
            df = pd.read_csv("synthetic_robot_logs.csv")
            if all(col in df.columns for col in ["packet_size", "interval", "is_botnet"]):
                st.success("✅ Loaded dataset from file")
                return df
            else:
                st.warning("⚠️ CSV missing required columns. Using synthetic data.")
                return create_sample_data(1000)
        except Exception as e:
            st.warning(f"⚠️ Error reading file: {e}. Using synthetic data.")
            return create_sample_data(1000)
    else:
        st.warning("⚠️ Dataset not found. Using synthetic data.")
        return create_sample_data(1000)

# --------------------
# Train Model
# --------------------
@st.cache_resource
def train_model(df):
    X = df[["packet_size", "interval"]]
    y = df["is_botnet"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

# --------------------
# Packet Generator
# --------------------
def generate_packet(df):
    row = df.sample(1).iloc[0]
    packet_size = np.random.normal(row["packet_size"], 5)
    interval = np.random.normal(row["interval"], 0.05)
    return abs(packet_size), abs(interval), int(row["is_botnet"])

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="Botnet Detection Simulation", layout="wide")
st.title("🤖 Real-Time Botnet Detection Simulation (From Real Dataset)")

# Sidebar first — always visible
st.sidebar.header("⚙️ Simulation Settings")
ticks = st.sidebar.number_input("Number of packets", min_value=5, max_value=200, value=20)
delay = st.sidebar.slider("Delay between packets (seconds)", min_value=0.1, max_value=3.0, value=1.0)
run_button = st.sidebar.button("Start Simulation")

# Load data AFTER UI
df = load_data()
model, scaler = train_model(df)

placeholder_table = st.empty()
placeholder_chart = st.empty()
df_sim = pd.DataFrame(columns=["Packet Size", "Interval", "Prediction", "Confidence", "Actual"])

if run_button:
    progress_bar = st.progress(0)
    for i in range(ticks):
        packet_size, interval, actual_label = generate_packet(df)
        X_new = scaler.transform([[packet_size, interval]])
        pred_prob = model.predict_proba(X_new)[0][1]
        pred_class = int(pred_prob > 0.5)
        confidence = pred_prob if pred_class == 1 else 1 - pred_prob

        status = "BOTNET ⚠️" if pred_class == 1 else "NORMAL ✅"

        df_sim.loc[len(df_sim)] = [
            round(packet_size, 2),
            round(interval, 2),
            status,
            round(confidence * 100, 2),
            "BOTNET" if actual_label == 1 else "NORMAL"
        ]

        placeholder_table.dataframe(df_sim.tail(10))
        chart_data = df_sim["Prediction"].value_counts().reset_index()
        chart_data.columns = ["Prediction", "Count"]
        placeholder_chart.bar_chart(chart_data.set_index("Prediction"))

        progress_bar.progress((i + 1) / ticks)
        time.sleep(delay)

    st.success("✅ Simulation complete!")

if not df_sim.empty:
    st.subheader("📊 Final Simulation Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Packets", len(df_sim))
    col2.metric("Detected Botnets", (df_sim["Prediction"] == "BOTNET ⚠️").sum())
    col3.metric("Detected Normal", (df_sim["Prediction"] == "NORMAL ✅").sum())

    st.subheader("🔍 Detailed Results")
    st.dataframe(df_sim)
