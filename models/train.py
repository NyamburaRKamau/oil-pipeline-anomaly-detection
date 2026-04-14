"""
Oil Pipeline Anomaly Detection - Model Training
Uses Isolation Forest for unsupervised anomaly detection
Tracks experiments with MLflow
"""

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generate_data import generate_pipeline_data


def train_model(contamination=0.05, n_estimators=100, random_state=42):
    """Train Isolation Forest model with MLflow tracking"""

    # Generate/load data
    print("Generating pipeline sensor data...")
    df = generate_pipeline_data(n_samples=5000, anomaly_ratio=contamination)

    # Features and labels
    feature_cols = ['pressure_psi', 'temperature_c', 'flow_rate_m3hr', 'vibration_mms']
    X = df[feature_cols].values
    y = df['anomaly'].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Start MLflow run
    mlflow.set_experiment("oil-pipeline-anomaly-detection")

    with mlflow.start_run(run_name="isolation-forest-v1"):

        # Log parameters
        mlflow.log_param("model_type", "IsolationForest")
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
        mlflow.log_param("features", feature_cols)

        # Train model
        print(f"\nTraining Isolation Forest (contamination={contamination}, n_estimators={n_estimators})...")
        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train_scaled)

        # Predictions (Isolation Forest returns -1 for anomaly, 1 for normal)
        y_pred_raw = model.predict(X_test_scaled)
        y_pred = np.where(y_pred_raw == -1, 1, 0)  # Convert to 0/1

        # Anomaly scores
        anomaly_scores = -model.score_samples(X_test_scaled)

        # Metrics
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly'])

        print("\n" + "="*50)
        print("MODEL PERFORMANCE")
        print("="*50)
        print(report)

        # Log metrics
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("anomaly_detection_rate",
                         (y_pred[y_test == 1] == 1).mean() if y_test.sum() > 0 else 0)

        # Log model
        mlflow.sklearn.log_model(model, "isolation_forest_model")

        # Save confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        ax.set_title('Confusion Matrix - Oil Pipeline Anomaly Detection')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        mlflow.log_artifact('confusion_matrix.png')
        plt.close()

        # Feature importance via anomaly score contribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        feature_names = ['Pressure (PSI)', 'Temperature (°C)', 'Flow Rate (m³/hr)', 'Vibration (mm/s)']
        colors = ['#c4673a', '#2c2418', '#8b6f5e', '#c9983a']

        for i, (ax, fname, color) in enumerate(zip(axes.flatten(), feature_names, colors)):
            normal_vals = X_test[y_test == 0, i]
            anomaly_vals = X_test[y_test == 1, i]
            ax.hist(normal_vals, bins=30, alpha=0.7, label='Normal', color='#4a6741')
            ax.hist(anomaly_vals, bins=30, alpha=0.7, label='Anomaly', color=color)
            ax.set_title(fname)
            ax.legend()
            ax.set_xlabel('Value')
            ax.set_ylabel('Count')

        plt.suptitle('Sensor Reading Distribution: Normal vs Anomaly', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('sensor_distributions.png', dpi=150, bbox_inches='tight')
        mlflow.log_artifact('sensor_distributions.png')
        plt.close()

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow Run ID: {run_id}")

    # Save model and scaler locally
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/anomaly_detector.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')

    print("\nModel saved to models/anomaly_detector.pkl")
    print("Scaler saved to models/scaler.pkl")

    return model, scaler, feature_cols


if __name__ == "__main__":
    model, scaler, feature_cols = train_model()
    print("\nTraining complete!")
