# IPL Win Predictor - Low Level Design Document

## 1. API Endpoint Specifications

### 1.1 Prediction Endpoint

#### POST /predict
**Purpose**: Make IPL match outcome predictions

**Request Body**:
```json
{
  "team1": "string (required)",
  "team2": "string (required)",
  "venue": "string (required)",
  "city": "string (required)",
  "team1_win_percentage": "float (required)",
  "team2_win_percentage": "float (required)",
  "team1_recent_form": "float (optional)",
  "team2_recent_form": "float (optional)",
  "team1_head_to_head": "float (optional)",
  "team2_head_to_head": "float (optional)"
}
```

**Response Body**:
```json
{
  "prediction": "integer (0 or 1)",
  "probability": "float (0.0 to 1.0)",
  "confidence": "string (Low/Medium/High)",
  "features_used": ["array of feature names"]
}
```

**Status Codes**:
- `200 OK`: Successful prediction
- `400 Bad Request`: Invalid input data
- `503 Service Unavailable`: Model not loaded

**Example Request**:
```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "team1": "Mumbai Indians",
    "team2": "Chennai Super Kings",
    "venue": "Wankhede Stadium",
    "city": "Mumbai",
    "team1_win_percentage": 0.65,
    "team2_win_percentage": 0.58,
    "team1_recent_form": 0.7,
    "team2_recent_form": 0.6,
    "team1_head_to_head": 0.55,
    "team2_head_to_head": 0.45
  }'
```

**Example Response**:
```json
{
  "prediction": 1,
  "probability": 0.75,
  "confidence": "High",
  "features_used": [
    "Team1_encoded",
    "Team2_encoded",
    "Venue_encoded",
    "City_encoded",
    "team1_win_percentage",
    "team2_win_percentage",
    "team1_recent_form",
    "team2_recent_form",
    "team1_head_to_head",
    "team2_head_to_head",
    "win_percentage_diff",
    "recent_form_diff",
    "h2h_advantage"
  ]
}
```

### 1.2 Health Check Endpoint

#### GET /health
**Purpose**: Check API service health

**Response Body**:
```json
{
  "status": "string (healthy/unhealthy)",
  "model_loaded": "boolean",
  "timestamp": "string (ISO 8601 format)"
}
```

**Status Codes**:
- `200 OK`: Service healthy
- `503 Service Unavailable`: Service unhealthy

### 1.3 Metrics Endpoint

#### GET /metrics
**Purpose**: Expose Prometheus metrics

**Response**: Prometheus-formatted metrics

**Status Codes**:
- `200 OK`: Metrics available

### 1.4 Model Information Endpoint

#### GET /model-info
**Purpose**: Get model metadata

**Response Body**:
```json
{
  "model_name": "string",
  "model_type": "string",
  "training_date": "string (ISO 8601 format)",
  "feature_count": "integer",
  "features": ["array of feature names"],
  "performance_metrics": {
    "accuracy": "float",
    "precision": "float",
    "recall": "float",
    "f1_score": "float"
  }
}
```

### 1.5 API Documentation Endpoint

#### GET /docs
**Purpose**: Interactive API documentation (Swagger UI)

## 2. Data Models

### 2.1 PredictionRequest Model
```python
class PredictionRequest(BaseModel):
    team1: str = Field(..., description="First team name")
    team2: str = Field(..., description="Second team name")
    venue: str = Field(..., description="Match venue")
    city: str = Field(..., description="Match city")
    team1_win_percentage: float = Field(..., ge=0.0, le=1.0, description="Team 1 win percentage")
    team2_win_percentage: float = Field(..., ge=0.0, le=1.0, description="Team 2 win percentage")
    team1_recent_form: Optional[float] = Field(None, ge=0.0, le=1.0, description="Team 1 recent form")
    team2_recent_form: Optional[float] = Field(None, ge=0.0, le=1.0, description="Team 2 recent form")
    team1_head_to_head: Optional[float] = Field(None, ge=0.0, le=1.0, description="Team 1 head-to-head record")
    team2_head_to_head: Optional[float] = Field(None, ge=0.0, le=1.0, description="Team 2 head-to-head record")
```

### 2.2 PredictionResponse Model
```python
class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Prediction (0 or 1)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Prediction probability")
    confidence: str = Field(..., description="Confidence level (Low/Medium/High)")
    features_used: List[str] = Field(..., description="List of features used")
```

## 3. Database Schema

### 3.1 MLflow Tracking Database
```sql
-- Experiments table
CREATE TABLE experiments (
    experiment_id INTEGER PRIMARY KEY,
    name VARCHAR(256) NOT NULL,
    artifact_location VARCHAR(256),
    lifecycle_stage VARCHAR(32)
);

-- Runs table
CREATE TABLE runs (
    run_uuid VARCHAR(32) PRIMARY KEY,
    name VARCHAR(250),
    source_type VARCHAR(32),
    source_name VARCHAR(500),
    entry_point_name VARCHAR(50),
    user_id VARCHAR(256),
    status VARCHAR(9),
    start_time BIGINT,
    end_time BIGINT,
    source_version VARCHAR(50),
    lifecycle_stage VARCHAR(28),
    artifact_uri VARCHAR(200),
    experiment_id INTEGER,
    deleted_time BIGINT
);

-- Parameters table
CREATE TABLE params (
    key VARCHAR(250) NOT NULL,
    value VARCHAR(250) NOT NULL,
    run_uuid VARCHAR(32) NOT NULL,
    PRIMARY KEY (key, run_uuid)
);

-- Metrics table
CREATE TABLE metrics (
    key VARCHAR(250) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    timestamp BIGINT NOT NULL,
    step BIGINT NOT NULL DEFAULT 0,
    run_uuid VARCHAR(32) NOT NULL,
    PRIMARY KEY (key, timestamp, step, run_uuid)
);
```

## 4. Configuration Management

### 4.1 Model Configuration (configs/model_config.yaml)
```yaml
model:
  name: "ipl_win_predictor"
  type: "xgboost"

hyperparameters:
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
  subsample: 0.8
  colsample_bytree: 0.8
  random_state: 42

training:
  test_size: 0.2
  validation_size: 0.1
  random_state: 42

features:
  target_column: "winner"
  categorical_features:
    - "Team1"
    - "Team2"
    - "City"
    - "Venue"
  numerical_features:
    - "team1_win_percentage"
    - "team2_win_percentage"
    - "team1_recent_form"
    - "team2_recent_form"
    - "team1_head_to_head"
    - "team2_head_to_head"

mlflow:
  experiment_name: "ipl_win_prediction"
  tracking_uri: "sqlite:///mlruns.db"

monitoring:
  prometheus_port: 8000
  metrics_interval: 60
```

### 4.2 DVC Pipeline Configuration (dvc.yaml)
```yaml
stages:
  data_processing:
    cmd: python scripts/ingest_and_process.py
    deps:
      - data/raw/matches.csv
      - data/raw/deliveries.csv
      - scripts/ingest_and_process.py
    outs:
      - data/processed/processed_ipl_data.parquet:
          persist: true
    metrics:
      - metrics/data_quality.json:
          persist: true
      - metrics/processing_stats.json:
          persist: true

  feature_engineering:
    cmd: python scripts/feature_engineering.py
    deps:
      - data/processed/processed_ipl_data.parquet
      - scripts/feature_engineering.py
    outs:
      - data/features/:
          persist: true
    metrics:
      - metrics/feature_importance.json:
          persist: true

  model_training:
    cmd: python scripts/train_model.py
    deps:
      - data/features/
      - scripts/train_model.py
      - configs/model_config.yaml
    outs:
      - models/:
          persist: true
    metrics:
      - metrics/model_performance.json:
          persist: true
      - mlruns/:
          persist: true

  model_evaluation:
    cmd: python scripts/evaluate_model.py
    deps:
      - models/
      - data/features/
      - scripts/evaluate_model.py
    metrics:
      - metrics/evaluation_results.json:
          persist: true
      - reports/model_evaluation.html:
          persist: true
```

## 5. Error Handling

### 5.1 HTTP Status Codes
- `200 OK`: Successful operation
- `400 Bad Request`: Invalid input data
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service unavailable

### 5.2 Error Response Format
```json
{
  "error": "string (error message)",
  "detail": "string (detailed error information)",
  "timestamp": "string (ISO 8601 format)",
  "request_id": "string (unique request identifier)"
}
```

### 5.3 Exception Handling
```python
class PredictionError(Exception):
    """Custom exception for prediction errors"""
    pass

class ModelNotLoadedError(Exception):
    """Exception when model is not loaded"""
    pass

class ValidationError(Exception):
    """Exception for data validation errors"""
    pass
```

## 6. Logging Strategy

### 6.1 Log Levels
- **DEBUG**: Detailed debugging information
- **INFO**: General information about program execution
- **WARNING**: Warning messages for potentially problematic situations
- **ERROR**: Error messages for serious problems
- **CRITICAL**: Critical errors that may prevent the program from running

### 6.2 Log Format
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
```

### 6.3 Structured Logging
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "logger": "api.main",
  "message": "Prediction request received",
  "request_id": "uuid",
  "user_id": "user123",
  "prediction_time_ms": 150
}
```

## 7. Security Implementation

### 7.1 Input Validation
- **Pydantic Models**: Automatic validation
- **Type Checking**: Runtime type validation
- **Range Validation**: Numeric value bounds
- **String Sanitization**: Prevent injection attacks

### 7.2 Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: PredictionRequest):
    # Implementation
```

### 7.3 CORS Configuration
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 8. Performance Optimization

### 8.1 Caching Strategy
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def load_model():
    # Model loading logic
```

### 8.2 Async Operations
```python
async def predict(request: PredictionRequest):
    # Async prediction logic
    return await model.predict_async(features)
```

### 8.3 Database Connection Pooling
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    "postgresql://user:pass@localhost/db",
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30
)
```

## 9. Testing Specifications

### 9.1 Unit Test Structure
```python
class TestPredictionAPI:
    def test_valid_prediction_request(self):
        # Test valid request

    def test_invalid_prediction_request(self):
        # Test invalid request

    def test_model_not_loaded(self):
        # Test when model is not available
```

### 9.2 Integration Test Structure
```python
class TestEndToEnd:
    def test_complete_prediction_flow(self):
        # Test complete prediction flow

    def test_error_handling(self):
        # Test error scenarios
```

### 9.3 Performance Test Structure
```python
class TestPerformance:
    def test_prediction_latency(self):
        # Test prediction response time

    def test_concurrent_requests(self):
        # Test multiple concurrent requests
```