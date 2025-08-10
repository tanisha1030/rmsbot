import os
import time
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib

# --------------------
# Step 1: Load or Generate Dataset
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

    df = pd.DataFrame({
        'packet_size': packet_sizes,
        'interval': intervals,
        'is_botnet': labels.astype(int)
    })
    return df.sample(frac=1).reset_index(drop=True)

def load_data():
    file_path = "synthetic_robot_logs.csv"
    if os.path.exists(file_path):
        print(f"[INFO] Loading dataset from {file_path}")
        return pd.read_csv(file_path)
    else:
        print("[WARNING] Dataset not found. Generating synthetic data...")
        return create_sample_data()

# --------------------
# Step 2: Train Model
# --------------------
def train_model(df):
    X = df[['packet_size', 'interval']].values
    y = df['is_botnet'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: weights[i] for i in range(len(weights))}

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    print("[INFO] Training model...")
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)

    print("\n[RESULT] Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n[RESULT] Classification Report:\n", classification_report(y_test, y_pred))
    print(f"[RESULT] ROC-AUC: {roc_auc_score(y_test, y_pred_probs):.4f}")

    # Save model and scaler
    model.save("botnet_detection_model.h5")
    joblib.dump(scaler, "scaler.pkl")
    print("[INFO] Model and scaler saved.")

    return model, scaler

# --------------------
# Step 3: Simulation
# --------------------
def generate_packet():
    """Simulate a packet with random features"""
    is_botnet = np.random.choice([0, 1], p=[0.7, 0.3])  # 70% normal, 30% botnet
    if is_botnet == 0:
        packet_size = np.random.normal(500, 150)
        interval = np.random.exponential(2.0)
    else:
        packet_size = np.random.normal(800, 100)
        interval = np.random.exponential(0.5)
    return abs(packet_size), abs(interval)

def run_simulation(model, scaler, ticks=20, delay=1):
    print("\n[SIMULATION] Starting live traffic simulation...\n")
    for t in range(ticks):
        packet_size, interval = generate_packet()
        X_new = scaler.transform([[packet_size, interval]])
        pred_prob = model.predict(X_new)[0][0]
        pred_class = int(pred_prob > 0.5)

        status = "BOTNET ⚠️" if pred_class == 1 else "NORMAL ✅"
        print(f"Tick {t+1:02d}: PacketSize={packet_size:.2f}, Interval={interval:.2f} -> {status} (Conf={pred_prob:.2%})")

        time.sleep(delay)

# --------------------
# Main Execution
# --------------------
if __name__ == "__main__":
    df = load_data()

    if os.path.exists("botnet_detection_model.h5") and os.path.exists("scaler.pkl"):
        print("[INFO] Loading existing model...")
        model = load_model("botnet_detection_model.h5")
        scaler = joblib.load("scaler.pkl")
    else:
        model, scaler = train_model(df)

    run_simulation(model, scaler, ticks=30, delay=1)
