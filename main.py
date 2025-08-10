import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --------------------
# Step 1: Load Dataset
# --------------------
def load_data():
    df = pd.read_csv("synthetic_robot_logs.csv")
    if not all(col in df.columns for col in ["packet_size", "interval", "is_botnet"]):
        raise ValueError("Dataset must contain 'packet_size', 'interval', 'is_botnet' columns")
    return df

# --------------------
# Step 2: Train Model
# --------------------
def train_model(df):
    X = df[["packet_size", "interval"]]
    y = df["is_botnet"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n[MODEL PERFORMANCE]")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model, scaler

# --------------------
# Step 3: Simulation Generator
# --------------------
def generate_packet(df):
    """Generate a packet based on real dataset distributions."""
    # Pick a random row from dataset
    row = df.sample(1).iloc[0]
    is_botnet = row["is_botnet"]
    packet_size = np.random.normal(row["packet_size"], 5)  # small variation
    interval = np.random.normal(row["interval"], 0.05)
    return abs(packet_size), abs(interval), int(is_botnet)

# --------------------
# Step 4: Run Simulation
# --------------------
def run_simulation(model, scaler, df, ticks=20, delay=1):
    print("\n[SIMULATION STARTED]\n")
    for t in range(ticks):
        packet_size, interval, actual_label = generate_packet(df)
        X_new = scaler.transform([[packet_size, interval]])
        pred_class = model.predict(X_new)[0]
        pred_prob = model.predict_proba(X_new)[0][pred_class]

        status = "BOTNET ⚠️" if pred_class == 1 else "NORMAL ✅"
        print(f"Tick {t+1:02d}: PacketSize={packet_size:.2f}, Interval={interval:.2f} "
              f"-> {status} (Conf={pred_prob:.2%}) | Actual: {'BOTNET' if actual_label else 'NORMAL'}")
        time.sleep(delay)

# --------------------
# Main Execution
# --------------------
if __name__ == "__main__":
    df = load_data()
    model, scaler = train_model(df)
    run_simulation(model, scaler, df, ticks=30, delay=1)
