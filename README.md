# 🛢️ Oil Pipeline Anomaly Detection System

Real-time ML-powered anomaly detection for oil pipeline sensor data. Detects pressure spikes, temperature anomalies, flow rate drops, and vibration irregularities that indicate potential leaks or equipment failures.

## 🏗️ Architecture

```
Pipeline Sensors → FastAPI → Isolation Forest Model → Anomaly Alert
                      ↓
              Prometheus Metrics → Grafana Dashboard
                      ↓
              MLflow Experiment Tracking
```

## 📊 Sensors Monitored

| Sensor | Normal Range | Anomaly Threshold |
|--------|-------------|-------------------|
| Pressure | 140-160 PSI | >190 or <110 PSI |
| Temperature | 54-66°C | >90°C |
| Flow Rate | 180-220 m³/hr | <100 m³/hr |
| Vibration | 0.3-0.7 mm/s | >1.5 mm/s |

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate data & train model
```bash
python data/generate_data.py
python models/train.py
```

### 3. Start API
```bash
uvicorn api.main:app --reload --port 8000
```

### 4. Full stack with Docker
```bash
docker-compose up --build
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health status |
| `/predict` | POST | Single reading prediction |
| `/predict/batch` | POST | Batch predictions |
| `/model/info` | GET | Model metadata |
| `/docs` | GET | Interactive API docs |

## 🔍 Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pressure_psi": 195.0,
    "temperature_c": 61.0,
    "flow_rate_m3hr": 202.0,
    "vibration_mms": 0.51,
    "pipeline_id": "PIPE-001"
  }'
```

## 📈 MLflow Tracking

```bash
mlflow ui --port 5000
```
Visit http://localhost:5000 to see experiment runs.

## 🐳 Monitoring Stack

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **API Docs**: http://localhost:8000/docs

## 🧠 Model

- **Algorithm**: Isolation Forest (unsupervised anomaly detection)
- **Features**: pressure, temperature, flow rate, vibration
- **Contamination**: 5% (expected anomaly rate)
- **Tracking**: MLflow experiment tracking with artifact logging

## 📁 Project Structure

```
oil-pipeline-anomaly-detection/
├── data/
│   └── generate_data.py      # Synthetic sensor data generator
├── models/
│   └── train.py              # Model training with MLflow
├── api/
│   └── main.py               # FastAPI serving application
├── monitoring/
│   └── prometheus.yml        # Prometheus config
├── docker-compose.yml        # Full stack deployment
├── Dockerfile                # API container
├── requirements.txt          # Dependencies
└── README.md
```

## 👩‍💻 Author

Rosemary Kamau | [github.com/roseemaryy](https://github.com/roseemaryy)
