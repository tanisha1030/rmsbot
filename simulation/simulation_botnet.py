import simpy
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Simulation parameters
NUM_ROBOTS = 10
SIM_TIME = 100   # seconds
BOTNET_RATIO = 0.3
PACKET_GEN_INTERVAL_NORMAL = (1.5, 3.0)
PACKET_GEN_INTERVAL_BOTNET = (0.1, 0.8)
PACKET_SIZE_NORMAL = (400, 600)
PACKET_SIZE_BOTNET = (700, 900)

traffic_data = []

def robot_traffic(env, robot_id, is_botnet):
    while True:
        if is_botnet:
            interval = np.random.uniform(*PACKET_GEN_INTERVAL_BOTNET)
            packet_size = np.random.normal(np.mean(PACKET_SIZE_BOTNET), 50)
        else:
            interval = np.random.uniform(*PACKET_GEN_INTERVAL_NORMAL)
            packet_size = np.random.normal(np.mean(PACKET_SIZE_NORMAL), 50)

        traffic_data.append({
            "time": env.now,
            "robot_id": robot_id,
            "packet_size": abs(packet_size),
            "interval": abs(interval),
            "is_botnet": int(is_botnet)
        })

        yield env.timeout(interval)

def run_simulation():
    env = simpy.Environment()
    botnet_ids = set(np.random.choice(range(NUM_ROBOTS), int(NUM_ROBOTS * BOTNET_RATIO), replace=False))

    for robot_id in range(NUM_ROBOTS):
        env.process(robot_traffic(env, robot_id, robot_id in botnet_ids))

    env.run(until=SIM_TIME)
    df = pd.DataFrame(traffic_data)
    return df

def train_model(df):
    X = df[['packet_size', 'interval']].values
    y = df['is_botnet'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Model Accuracy on simulated data: {acc:.2f}")
    return model, scaler

if __name__ == "__main__":
    simulated_df = run_simulation()
    print(simulated_df.head())
    simulated_df.to_csv("simulated_robot_logs.csv", index=False)
    print("Simulated dataset saved as simulated_robot_logs.csv")
    train_model(simulated_df)
