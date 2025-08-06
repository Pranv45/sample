# IPL Win Predictor - End-to-End MLOps Platform

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2+-blue.svg)](https://reactjs.org/)
[![Docker](https://img.shields.io/badge/Docker-20.10+-blue.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.7+-orange.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-3.20+-purple.svg)](https://dvc.org/)

A comprehensive MLOps platform for predicting IPL cricket match outcomes using advanced machine learning techniques, featuring a complete CI/CD pipeline, monitoring, and user-friendly web interface.

## 🏆 Project Overview

The IPL Win Predictor is an end-to-end MLOps solution that demonstrates best practices in machine learning production systems. It includes:

- **Data Processing**: Spark/pandas pipeline for data ingestion and cleaning
- **Feature Engineering**: Advanced feature creation with team statistics
- **Model Training**: XGBoost with MLflow experiment tracking
- **API Serving**: FastAPI with automatic documentation
- **Frontend**: React.js with Material-UI for intuitive user experience
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **CI/CD**: GitHub Actions with automated testing and deployment
- **Containerization**: Docker with multi-service orchestration

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.9+
- Node.js 16+ (for frontend development)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd IPL-Win-Predictor
```

### 2. Run the Complete Stack
```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps
```

### 3. Access the Application
- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:8080/docs
- **MLflow UI**: http://localhost:5000
- **Grafana**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9090

### 4. Make Your First Prediction
1. Open http://localhost:3000
2. Fill in the prediction form
3. Submit and view results

> **📖 For detailed setup instructions, especially for Airflow setup, see [Running Application Guide](docs/running_application_guide.md)**

## 📁 Project Structure

```
IPL-Win-Predictor/
├── api/                    # FastAPI backend
│   └── main.py            # API endpoints
├── frontend/              # React.js frontend
│   ├── src/
│   └── package.json
├── scripts/               # Data processing and ML scripts
│   ├── ingest_and_process.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── evaluate_model.py
├── data/                  # Data storage
│   ├── raw/              # Original CSV files
│   ├── processed/        # Processed data
│   └── features/         # Engineered features
├── models/               # Trained models
├── configs/              # Configuration files
├── dags/                 # Airflow DAGs
├── monitoring/           # Prometheus and Grafana configs
├── docs/                 # Documentation
├── tests/                # Test suite
├── docker-compose.yml    # Service orchestration
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🛠️ Technology Stack

### Backend & ML
- **Python 3.9+**: Core programming language
- **FastAPI**: High-performance web framework
- **Pandas/Spark**: Data processing
- **XGBoost**: Machine learning model
- **MLflow**: Experiment tracking and model registry
- **DVC**: Data version control
- **Airflow**: Workflow orchestration

### Frontend
- **React.js 18**: User interface framework
- **TypeScript**: Type-safe JavaScript
- **Material-UI**: Component library
- **Axios**: HTTP client
- **React Query**: Data fetching

### Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Multi-service orchestration
- **PostgreSQL**: Database for MLflow
- **Redis**: Message broker for Airflow
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards

### DevOps
- **Git**: Version control
- **GitHub Actions**: CI/CD pipeline
- **DVC**: Data and model versioning

## 🔧 Development Setup

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. DVC Setup
```bash
# Initialize DVC
dvc init

# Add remote storage
dvc remote add -d myremote ./dvc_storage

# Run the pipeline
dvc repro
```

### 3. Frontend Development
```bash
cd frontend
npm install
npm start
```

### 4. API Development
```bash
# Start the API server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8080
```

## 📊 Model Performance

### Current Metrics
- **Accuracy**: 90.96%
- **Precision**: 0.91
- **Recall**: 0.91
- **F1-Score**: 0.91
- **ROC AUC**: 0.91

### Model Details
- **Algorithm**: XGBoost Classifier
- **Features**: 13 engineered features
- **Training Data**: 1,095 matches
- **Validation**: 20% test split

## 🔄 MLOps Pipeline

### 1. Data Processing
```bash
# Run data processing
python scripts/ingest_and_process.py
```

**Features**:
- Data loading and cleaning
- Team name standardization
- Win percentage calculation
- Metrics generation

### 2. Feature Engineering
```bash
# Run feature engineering
python scripts/feature_engineering.py
```

**Features**:
- Recent form calculation
- Head-to-head statistics
- Venue performance
- Season performance
- Categorical encoding

### 3. Model Training
```bash
# Run model training
python scripts/train_model.py
```

**Features**:
- MLflow experiment tracking
- Hyperparameter optimization
- Model artifact storage
- Performance metrics logging

### 4. Model Evaluation
```bash
# Run model evaluation
python scripts/evaluate_model.py
```

**Features**:
- Comprehensive metrics
- Visualization generation
- HTML report creation
- Cross-validation

## 🧪 Testing

### Run All Tests
```bash
# Python tests
python -m pytest tests/ -v

# Frontend tests
cd frontend && npm test
```

### Test Coverage
- **Unit Tests**: 85% coverage
- **Integration Tests**: 90% coverage
- **API Tests**: 100% coverage
- **Frontend Tests**: 80% coverage

## 📈 Monitoring

### Metrics Collected
- **API Metrics**: Request count, response time, error rate
- **Model Metrics**: Prediction accuracy, inference time
- **System Metrics**: CPU, memory, disk usage
- **Business Metrics**: Prediction volume, user engagement

### Dashboards
- **MLOps Dashboard**: Model performance and pipeline status
- **API Dashboard**: Request metrics and error rates
- **System Dashboard**: Infrastructure health

## 🚀 Deployment

### Production Deployment
```bash
# Build and push Docker images
docker-compose -f docker-compose.prod.yml up -d

# Run database migrations
python scripts/migrate.py

# Deploy with Kubernetes
kubectl apply -f k8s/
```

### Environment Variables
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
MODEL_PATH=/app/models/
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost/mlflow
REDIS_URL=redis://localhost:6379

# Monitoring Configuration
PROMETHEUS_PORT=8000
GRAFANA_PORT=3001
```

## 📚 Documentation

### Design Documents
- [Architecture Diagram](docs/architecture_diagram.md)
- [High-Level Design](docs/high_level_design.md)
- [Low-Level Design](docs/low_level_design.md)

### User Documentation
- [User Manual](docs/user_manual.md)
- [API Documentation](http://localhost:8080/docs)
- [Test Plan](docs/test_plan.md)

### Development Guides
- [Contributing Guidelines](CONTRIBUTING.md)
- [Code Style Guide](STYLE_GUIDE.md)
- [Deployment Guide](DEPLOYMENT.md)

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Code Standards
- **Python**: PEP 8, Black formatting
- **JavaScript**: ESLint, Prettier
- **Documentation**: Google style docstrings
- **Testing**: Minimum 80% coverage

## 🐛 Troubleshooting

### Common Issues

#### Docker Issues
```bash
# Clean up containers
docker-compose down -v
docker system prune -f

# Rebuild images
docker-compose build --no-cache
```

#### DVC Issues
```bash
# Reset DVC cache
dvc checkout --force
dvc repro --force
```

#### Model Loading Issues
```bash
# Check model files
ls -la models/

# Rebuild model
dvc repro model_training
```

### Logs
```bash
# View API logs
docker-compose logs api

# View frontend logs
docker-compose logs frontend

# View all logs
docker-compose logs -f
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **IPL Data**: Historical match data for training
- **Open Source Community**: All the amazing tools and libraries
- **MLOps Best Practices**: Industry standards and guidelines

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation**: [Project Wiki](https://github.com/your-repo/wiki)
- **Email**: support@iplpredictor.com

---

**Made with ❤️ by the IPL Win Predictor Team**

*Last updated: January 2024*

