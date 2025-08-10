import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Import simulation
from simulation.simulation_botnet import run_simulation

# Configure the page
st.set_page_config(page_title="Botnet Detection", layout="wide")
st.title("🤖 Botnet Detection in Robotic Network Logs (Simulation-Driven)")

# Sidebar: Simulation Controls
st.sidebar.header("⚡ Simulation Control")
if st.sidebar.button("Run Simulation"):
    st.info("🚀 Running network traffic simulation...")
    df = run_simulation()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/simulated_robot_logs.csv", index=False)
    st.success("✅ Simulation complete! Data updated.")
    st.experimental_rerun()

# Create sample data if needed
def create_sample_data():
    np.random.seed(42)
    n_samples = 1000
    normal_packet_sizes = np.random.normal(500, 150, n_samples//2)
    normal_intervals = np.random.exponential(2.0, n_samples//2)
    botnet_packet_sizes = np.random.normal(800, 100, n_samples//2)
    botnet_intervals = np.random.exponential(0.5, n_samples//2)
    packet_sizes = np.concatenate([normal_packet_sizes, botnet_packet_sizes])
    intervals = np.concatenate([normal_intervals, botnet_intervals])
    labels = np.concatenate([np.zeros(n_samples//2), np.ones(n_samples//2)])
    packet_sizes = np.abs(packet_sizes)
    intervals = np.abs(intervals)
    df = pd.DataFrame({
        'packet_size': packet_sizes,
        'interval': intervals,
        'is_botnet': labels.astype(int)
    })
    return df.sample(frac=1).reset_index(drop=True)

# Load data
@st.cache_data
def load_data():
    try:
        if os.path.exists("data/simulated_robot_logs.csv"):
            df = pd.read_csv("data/simulated_robot_logs.csv")
            st.success("✅ Loaded simulated dataset.")
            return df
        elif os.path.exists("data/synthetic_robot_logs.csv"):
            df = pd.read_csv("data/synthetic_robot_logs.csv")
            st.info("ℹ️ Loaded default dataset.")
            return df
        else:
            st.warning("⚠️ No dataset found. Using sample data.")
            return create_sample_data()
    except:
        return create_sample_data()

# Train model
@st.cache_resource
def train_model():
    df = load_data()
    if df is not None and 'is_botnet' in df.columns:
        X = df[['packet_size', 'interval']].values
        y = df['is_botnet'].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        model.fit(X_train_scaled, y_train)
        train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
        test_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
        st.info(f"📊 Training Accuracy: {train_accuracy:.3f} | Test Accuracy: {test_accuracy:.3f}")
        return model, scaler, test_accuracy
    return None, None, 0.0

# Load data and train
df = load_data()
model, scaler, model_accuracy = train_model()

if df is not None and model is not None:
    st.subheader("📊 Dataset Information")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Botnet Records", int(df['is_botnet'].sum()))
    with col3:
        st.metric("Normal Records", int(len(df) - df['is_botnet'].sum()))
    with col4:
        st.metric("Model Accuracy", f"{model_accuracy:.1%}")

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(10))

    # Prediction interface
    required_columns = ['packet_size', 'interval']
    if all(col in df.columns for col in required_columns):
        st.subheader("🔮 Predict Botnet Activity")
        min_packet, max_packet = df['packet_size'].min(), df['packet_size'].max()
        min_interval, max_interval = df['interval'].min(), df['interval'].max()
        col1, col2 = st.columns(2)
        with col1:
            packet_size = st.number_input(
                f"Packet Size ({min_packet:.1f} - {max_packet:.1f})", 
                min_value=0.0, max_value=max_packet*2,
                value=float(df['packet_size'].mean()), step=10.0
            )
        with col2:
            interval = st.number_input(
                f"Interval ({min_interval:.2f} - {max_interval:.2f})", 
                min_value=0.0, max_value=max_interval*2,
                value=float(df['interval'].mean()), step=0.1
            )
        if st.button("🔮 Predict", type="primary"):
            input_data = np.array([[packet_size, interval]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            if prediction == 1:
                st.error(f"🔴 Botnet Activity Detected! Confidence: {max(probability):.1%}")
            else:
                st.success(f"🟢 Normal Activity. Confidence: {max(probability):.1%}")

    # Visualizations
    st.subheader("📊 Data Visualization")
    tab1, tab2, tab3 = st.tabs(["🎯 Scatter Plot", "📈 Distributions", "⚖️ Class Balance"])
    with tab1:
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['#2E8B57', '#DC143C']
        labels = ['Normal', 'Botnet']
        for i, label in enumerate(labels):
            mask = df['is_botnet'] == i
            ax.scatter(df[mask]['packet_size'], df[mask]['interval'], 
                       alpha=0.6, s=50, c=colors[i], label=label)
        ax.set_xlabel("Packet Size (bytes)")
        ax.set_ylabel("Interval (seconds)")
        ax.legend()
        st.pyplot(fig)
    with tab2:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.histplot(df[df['is_botnet']==0]['packet_size'], ax=axes[0], color="green", label="Normal", kde=True)
        sns.histplot(df[df['is_botnet']==1]['packet_size'], ax=axes[0], color="red", label="Botnet", kde=True)
        axes[0].set_title("Packet Size Distribution")
        sns.histplot(df[df['is_botnet']==0]['interval'], ax=axes[1], color="green", label="Normal", kde=True)
        sns.histplot(df[df['is_botnet']==1]['interval'], ax=axes[1], color="red", label="Botnet", kde=True)
        axes[1].set_title("Interval Distribution")
        st.pyplot(fig)
    with tab3:
        class_counts = df['is_botnet'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(['Normal', 'Botnet'], class_counts, color=['green', 'red'])
        st.pyplot(fig)
else:
    st.error("❌ Unable to load dataset or train model.")
