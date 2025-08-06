# IPL Win Predictor - Application Running Guide

This guide provides step-by-step instructions to run the complete IPL Win Predictor MLOps application. The application consists of multiple components that can be run individually or together using Docker Compose.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start (Docker Compose)](#quick-start-docker-compose)
3. [Individual Component Setup](#individual-component-setup)
4. [Troubleshooting](#troubleshooting)
5. [Accessing the Application](#accessing-the-application)

## Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS, or Linux
- **RAM**: Minimum 8GB (16GB recommended for full stack)
- **Storage**: At least 5GB free space
- **Docker**: Docker Desktop installed and running
- **Git**: Git installed for version control

### Software Installation

#### 1. Docker Desktop
```bash
# Download and install Docker Desktop from:
# https://www.docker.com/products/docker-desktop/

# Verify installation
docker --version
docker-compose --version
```

#### 2. Git
```bash
# Windows: Download from https://git-scm.com/download/win
# macOS: brew install git
# Linux: sudo apt-get install git

# Verify installation
git --version
```

#### 3. Python (for local development)
```bash
# Download Python 3.8+ from https://www.python.org/downloads/
# Verify installation
python --version
pip --version
```

## Quick Start (Docker Compose)

The easiest way to run the entire application is using Docker Compose, which will start all services automatically.

### Step 1: Clone and Navigate
```bash
# Clone the repository (if not already done)
git clone <your-repository-url>
cd IPL-Win-Predictor

# Or if you're already in the project directory
cd IPL-Win-Predictor
```

### Step 2: Run the Complete Stack
```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps
```

### Step 3: Verify Services
```bash
# Check if all containers are running
docker-compose logs

# Check specific service logs
docker-compose logs airflow
docker-compose logs mlflow
docker-compose logs api
```

### Step 4: Access the Application
Once all services are running, you can access:

- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **MLflow**: http://localhost:5000
- **Airflow**: http://localhost:8080 (admin/admin)
- **Grafana**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9090

## Individual Component Setup

If you prefer to run components individually or troubleshoot specific services, follow these steps:

### 1. Data Processing Pipeline

#### Using DVC (Recommended)
```bash
# Install DVC
pip install dvc

# Initialize DVC (if not already done)
dvc init

# Run the data processing pipeline
dvc repro

# Check the results
ls data/processed/
ls metrics/
```

#### Manual Data Processing
```bash
# Run data ingestion and processing
python scripts/ingest_and_process.py

# Run feature engineering
python scripts/feature_engineering.py

# Check outputs
ls data/processed/
```

### 2. Model Training

#### Using MLflow
```bash
# Start MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000

# In another terminal, run model training
python scripts/train_model.py

# Check MLflow UI at http://localhost:5000
```

#### Manual Training
```bash
# Run training script
python scripts/train_model.py

# Check model files
ls models/
ls metrics/
```

### 3. FastAPI Backend

#### Using Docker
```bash
# Build and run the API container
docker build -f Dockerfile.api -t ipl-predictor-api .
docker run -p 8000:8000 ipl-predictor-api
```

#### Local Development
```bash
# Install dependencies
pip install fastapi uvicorn

# Run the API server
python api/main.py

# Or using uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. React Frontend

#### Using Docker
```bash
# Build and run the frontend container
docker build -f Dockerfile.frontend -t ipl-predictor-frontend .
docker run -p 3000:3000 ipl-predictor-frontend
```

#### Local Development
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

### 5. Airflow Setup (The Tricky Part)

Airflow can be complex to set up, especially on Windows. Here's a detailed guide:

#### Using Docker Compose (Recommended)
```bash
# The docker-compose.yaml already includes Airflow configuration
docker-compose up airflow-webserver airflow-scheduler -d

# Check Airflow status
docker-compose logs airflow-webserver
docker-compose logs airflow-scheduler
```

#### Manual Airflow Setup (Advanced)
```bash
# Install Airflow
pip install apache-airflow

# Set Airflow home
export AIRFLOW_HOME=./airflow

# Initialize Airflow database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Start Airflow webserver
airflow webserver --port 8080

# In another terminal, start scheduler
airflow scheduler
```

#### Airflow DAG Setup
```bash
# Copy DAG files to Airflow DAGs folder
cp dags/*.py ./airflow/dags/

# Or if using Docker, the DAGs are mounted automatically
# Check if DAGs are loaded
docker-compose exec airflow-webserver airflow dags list
```

### 6. Monitoring Stack

#### Prometheus & Grafana
```bash
# Start monitoring services
docker-compose up prometheus grafana -d

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Access Grafana at http://localhost:3001
# Default credentials: admin/admin
```

#### Manual Prometheus Setup
```bash
# Download Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.37.0/prometheus-2.37.0.windows-amd64.zip
unzip prometheus-2.37.0.windows-amd64.zip

# Start Prometheus
./prometheus.exe --config.file=monitoring/prometheus.yml
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Docker Issues
```bash
# If Docker containers fail to start
docker-compose down
docker system prune -f
docker-compose up -d

# Check container logs
docker-compose logs [service-name]
```

#### 2. Airflow Issues
```bash
# Reset Airflow database
docker-compose down
docker volume rm ipl-win-predictor_airflow-db
docker-compose up airflow-webserver airflow-scheduler -d

# Check DAG status
docker-compose exec airflow-webserver airflow dags list
docker-compose exec airflow-webserver airflow dags test ipl_ml_pipeline 2024-01-01
```

#### 3. Port Conflicts
```bash
# Check which processes are using ports
netstat -ano | findstr :8080  # Windows
lsof -i :8080                 # macOS/Linux

# Kill process if needed
taskkill /PID [process-id] /F  # Windows
kill -9 [process-id]           # macOS/Linux
```

#### 4. Memory Issues
```bash
# Increase Docker memory limit in Docker Desktop settings
# Recommended: 8GB+ for full stack

# Check memory usage
docker stats
```

#### 5. DVC Issues
```bash
# Reset DVC cache
dvc gc -f

# Re-run pipeline
dvc repro --force

# Check DVC status
dvc status
```

### Windows-Specific Issues

#### 1. File Permission Errors
```bash
# Run PowerShell as Administrator
# Or use Git Bash for better Unix-like environment
```

#### 2. Path Issues
```bash
# Use forward slashes in paths
# Or escape backslashes properly
```

#### 3. Line Ending Issues
```bash
# Configure Git for Windows
git config --global core.autocrlf true
```

## Accessing the Application

### 1. Frontend Application
- **URL**: http://localhost:3000
- **Features**:
  - IPL match prediction form
  - Real-time predictions
  - Historical data visualization

### 2. API Documentation
- **URL**: http://localhost:8000/docs
- **Features**:
  - Interactive API documentation
  - Test endpoints directly
  - View request/response schemas

### 3. MLflow Tracking
- **URL**: http://localhost:5000
- **Features**:
  - View model experiments
  - Compare model performance
  - Download model artifacts

### 4. Airflow Dashboard
- **URL**: http://localhost:8080
- **Credentials**: admin/admin
- **Features**:
  - View DAG runs
  - Monitor pipeline status
  - Trigger manual runs

### 5. Grafana Dashboards
- **URL**: http://localhost:3001
- **Credentials**: admin/admin
- **Features**:
  - Real-time metrics visualization
  - Custom dashboards
  - Alerting

### 6. Prometheus
- **URL**: http://localhost:9090
- **Features**:
  - Raw metrics data
  - Query interface
  - Target status

## Testing the Application

### 1. API Testing
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test prediction endpoint
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "team1": "Mumbai Indians",
    "team2": "Chennai Super Kings",
    "venue": "Wankhede Stadium",
    "city": "Mumbai"
  }'
```

### 2. Frontend Testing
1. Open http://localhost:3000
2. Fill in the prediction form
3. Submit and verify results
4. Test different team combinations

### 3. Pipeline Testing
```bash
# Test DVC pipeline
dvc repro

# Test Airflow DAG
docker-compose exec airflow-webserver airflow dags test ipl_ml_pipeline 2024-01-01
```

## Stopping the Application

### Stop All Services
```bash
# Stop all Docker containers
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

### Stop Individual Services
```bash
# Stop specific service
docker-compose stop [service-name]

# Stop multiple services
docker-compose stop airflow-webserver airflow-scheduler
```

## Development Workflow

### 1. Making Changes
```bash
# Edit code files
# Test changes locally
python scripts/train_model.py

# Commit changes
git add .
git commit -m "Update model training"

# Push to repository
git push origin main
```

### 2. CI/CD Pipeline
- Changes pushed to main branch trigger GitHub Actions
- Automated testing and deployment
- Check GitHub Actions tab for status

### 3. Model Updates
```bash
# Retrain model
python scripts/train_model.py

# Update DVC
dvc add models/
dvc push

# Commit changes
git add .
git commit -m "Update model"
git push
```

## Performance Optimization

### 1. Resource Allocation
- **Docker Desktop**: Allocate 8GB+ RAM
- **CPU**: 4+ cores recommended
- **Storage**: SSD preferred for faster I/O

### 2. Service Scaling
```bash
# Scale specific services
docker-compose up --scale api=3

# Monitor resource usage
docker stats
```

### 3. Caching
- Enable Docker layer caching
- Use DVC for data caching
- Implement API response caching

## Security Considerations

### 1. Production Deployment
- Change default passwords
- Use environment variables for secrets
- Enable HTTPS
- Implement proper authentication

### 2. Network Security
- Use internal Docker networks
- Restrict external access
- Monitor network traffic

## Support and Maintenance

### 1. Logs and Monitoring
```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f api

# Check service health
docker-compose ps
```

### 2. Backup and Recovery
```bash
# Backup data
docker-compose exec postgres pg_dump -U airflow > backup.sql

# Restore data
docker-compose exec -T postgres psql -U airflow < backup.sql
```

### 3. Updates and Maintenance
```bash
# Update Docker images
docker-compose pull
docker-compose up -d

# Update dependencies
pip install --upgrade -r requirements.txt
```

This guide should help you successfully run the IPL Win Predictor application. If you encounter any specific issues, refer to the troubleshooting section or check the logs for detailed error messages.