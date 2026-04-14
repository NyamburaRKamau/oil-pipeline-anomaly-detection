"""
Oil Pipeline Sensor Data Generator
Generates synthetic sensor readings with injected anomalies
Sensors: pressure, temperature, flow_rate, vibration
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_pipeline_data(n_samples=5000, anomaly_ratio=0.05, seed=42):
    np.random.seed(seed)

    timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i * 5) for i in range(n_samples)]

    # Normal sensor readings
    pressure = np.random.normal(loc=150, scale=5, size=n_samples)       # PSI
    temperature = np.random.normal(loc=60, scale=3, size=n_samples)      # Celsius
    flow_rate = np.random.normal(loc=200, scale=10, size=n_samples)      # m3/hr
    vibration = np.random.normal(loc=0.5, scale=0.1, size=n_samples)    # mm/s

    labels = np.zeros(n_samples, dtype=int)

    # Inject anomalies
    n_anomalies = int(n_samples * anomaly_ratio)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)

    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['pressure_spike', 'pressure_drop', 'temp_spike', 'flow_drop', 'vibration_spike'])

        if anomaly_type == 'pressure_spike':
            pressure[idx] = np.random.uniform(190, 220)
        elif anomaly_type == 'pressure_drop':
            pressure[idx] = np.random.uniform(80, 110)
        elif anomaly_type == 'temp_spike':
            temperature[idx] = np.random.uniform(90, 120)
        elif anomaly_type == 'flow_drop':
            flow_rate[idx] = np.random.uniform(50, 100)
        elif anomaly_type == 'vibration_spike':
            vibration[idx] = np.random.uniform(1.5, 3.0)

        labels[idx] = 1

    df = pd.DataFrame({
        'timestamp': timestamps,
        'pressure_psi': np.round(pressure, 2),
        'temperature_c': np.round(temperature, 2),
        'flow_rate_m3hr': np.round(flow_rate, 2),
        'vibration_mms': np.round(vibration, 3),
        'anomaly': labels
    })

    print(f"Generated {n_samples} samples")
    print(f"Normal readings: {(labels == 0).sum()}")
    print(f"Anomalies injected: {(labels == 1).sum()}")
    print(f"Anomaly ratio: {labels.mean():.2%}")

    return df


if __name__ == "__main__":
    df = generate_pipeline_data()
    df.to_csv("pipeline_sensor_data.csv", index=False)
    print("\nSample data:")
    print(df.head(10))
    print("\nData saved to pipeline_sensor_data.csv")
