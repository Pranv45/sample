# IPL Win Predictor - System Architecture

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           IPL Win Predictor MLOps Platform                        │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend UI   │    │   FastAPI API   │    │   ML Pipeline   │    │   Monitoring    │
│   (React/Vue)   │◄──►│   (Port 8080)   │◄──►│   (DVC/Airflow) │◄──►│ (Prometheus/    │
│                 │    │                 │    │                 │    │   Grafana)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         │                       │                       │                       │
         ▼                       ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Docker Compose│    │   Model Store   │    │   Data Store    │    │   MLflow UI     │
│   Orchestration │    │   (MLflow)      │    │   (DVC)         │    │   (Port 5000)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Component Details

### 1. Frontend Layer
- **Technology**: React.js with TypeScript
- **Purpose**: User interface for IPL win predictions
- **Features**:
  - Match prediction form
  - Results visualization
  - Pipeline monitoring dashboard
  - Model performance metrics

### 2. API Layer
- **Technology**: FastAPI with Pydantic
- **Endpoints**:
  - `/predict` - Make predictions
  - `/health` - Health check
  - `/metrics` - Prometheus metrics
  - `/model-info` - Model information
- **Port**: 8080

### 3. ML Pipeline Layer
- **Data Processing**: Pandas/Spark for data ingestion and cleaning
- **Feature Engineering**: Advanced feature creation with team statistics
- **Model Training**: XGBoost with MLflow tracking
- **Orchestration**: DVC pipeline with Airflow DAGs
- **Version Control**: Git + DVC for code, data, and model versioning

### 4. Monitoring Layer
- **Metrics Collection**: Prometheus client
- **Visualization**: Grafana dashboards
- **Logging**: Structured logging with different levels
- **Alerting**: Configurable alerts for model performance

### 5. Infrastructure Layer
- **Containerization**: Docker with multi-stage builds
- **Orchestration**: Docker Compose for local development
- **Database**: PostgreSQL for MLflow metadata
- **Message Queue**: Redis for Airflow tasks

## Data Flow

```
Raw Data (CSV) → Data Processing → Feature Engineering → Model Training → Model Evaluation → Model Deployment → API Serving → Frontend Display
     ↓              ↓                    ↓                    ↓                    ↓                    ↓                    ↓
   DVC Track    DVC Track          DVC Track          MLflow Track          Reports            Docker Image         User Interface
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend | React.js + TypeScript | User interface |
| API | FastAPI + Pydantic | REST API |
| ML Pipeline | DVC + Airflow | Pipeline orchestration |
| Model Training | XGBoost + MLflow | Model development |
| Monitoring | Prometheus + Grafana | Observability |
| Containerization | Docker + Docker Compose | Deployment |
| Version Control | Git + DVC | Code/Data/Model versioning |
| CI/CD | GitHub Actions | Automation |

## Security & Performance

- **API Security**: Input validation, rate limiting
- **Model Security**: Model versioning, A/B testing capability
- **Performance**: Async API calls, caching, load balancing ready
- **Scalability**: Microservices architecture, horizontal scaling capability