from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ipl_ml_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline for IPL Win Predictor',
    schedule_interval='@daily',
    catchup=False,
    tags=['mlops', 'ipl', 'prediction'],
)

def check_data_quality():
    """Check data quality before processing."""
    import pandas as pd
    import logging

    # Check raw data
    matches_df = pd.read_csv('/opt/airflow/data/raw/matches.csv')
    deliveries_df = pd.read_csv('/opt/airflow/data/raw/deliveries.csv')

    # Basic quality checks
    quality_metrics = {
        'matches_count': len(matches_df),
        'deliveries_count': len(deliveries_df),
        'matches_missing_values': matches_df.isnull().sum().sum(),
        'deliveries_missing_values': deliveries_df.isnull().sum().sum(),
        'timestamp': datetime.now().isoformat()
    }

    # Save quality metrics
    import json
    os.makedirs('/opt/airflow/metrics', exist_ok=True)
    with open('/opt/airflow/metrics/data_quality.json', 'w') as f:
        json.dump(quality_metrics, f, indent=2)

    logging.info(f"Data quality check completed: {quality_metrics}")
    return quality_metrics

def run_dvc_pipeline():
    """Run DVC pipeline stages."""
    import subprocess
    import logging

    # Run DVC pipeline
    try:
        # Data processing
        subprocess.run(['dvc', 'repro', 'data_processing'], check=True, cwd='/opt/airflow')
        logging.info("Data processing stage completed")

        # Feature engineering
        subprocess.run(['dvc', 'repro', 'feature_engineering'], check=True, cwd='/opt/airflow')
        logging.info("Feature engineering stage completed")

        # Model training
        subprocess.run(['dvc', 'repro', 'model_training'], check=True, cwd='/opt/airflow')
        logging.info("Model training stage completed")

        # Model evaluation
        subprocess.run(['dvc', 'repro', 'model_evaluation'], check=True, cwd='/opt/airflow')
        logging.info("Model evaluation stage completed")

    except subprocess.CalledProcessError as e:
        logging.error(f"DVC pipeline failed: {e}")
        raise

def deploy_model():
    """Deploy the trained model."""
    import shutil
    import logging

    # Copy model to deployment directory
    model_source = '/opt/airflow/models/ipl_win_predictor.pkl'
    model_dest = '/opt/airflow/models/deployed/ipl_win_predictor.pkl'

    os.makedirs('/opt/airflow/models/deployed', exist_ok=True)
    shutil.copy2(model_source, model_dest)

    # Copy scaler and model info
    shutil.copy2('/opt/airflow/models/scaler.pkl', '/opt/airflow/models/deployed/scaler.pkl')
    shutil.copy2('/opt/airflow/models/model_info.json', '/opt/airflow/models/deployed/model_info.json')

    logging.info("Model deployed successfully")
    return "Model deployed"

def send_notification():
    """Send notification about pipeline completion."""
    import logging
    logging.info("ML Pipeline completed successfully!")
    return "Pipeline completed"

# Define tasks
check_data_task = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    dag=dag,
)

data_processing_task = DockerOperator(
    task_id='data_processing',
    image='spark-env:latest',
    command='spark-submit /app/scripts/ingest_and_process.py',
    docker_url='unix://var/run/docker.sock',
    network_mode='bridge',
    volumes=[
        '/opt/airflow/data:/app/data',
        '/opt/airflow/scripts:/app/scripts'
    ],
    dag=dag,
)

def run_feature_engineering():
    """Run feature engineering script."""
    import subprocess
    import logging
    
    try:
        result = subprocess.run(['python', '/opt/airflow/scripts/feature_engineering.py'], 
                              check=True, capture_output=True, text=True)
        logging.info(f"Feature engineering completed: {result.stdout}")
        return "Feature engineering completed"
    except subprocess.CalledProcessError as e:
        logging.error(f"Feature engineering failed: {e.stderr}")
        raise

def run_model_training():
    """Run model training script."""
    import subprocess
    import logging
    
    try:
        result = subprocess.run(['python', '/opt/airflow/scripts/train_model.py'], 
                              check=True, capture_output=True, text=True)
        logging.info(f"Model training completed: {result.stdout}")
        return "Model training completed"
    except subprocess.CalledProcessError as e:
        logging.error(f"Model training failed: {e.stderr}")
        raise

def run_model_evaluation():
    """Run model evaluation script."""
    import subprocess
    import logging
    
    try:
        result = subprocess.run(['python', '/opt/airflow/scripts/evaluate_model.py'], 
                              check=True, capture_output=True, text=True)
        logging.info(f"Model evaluation completed: {result.stdout}")
        return "Model evaluation completed"
    except subprocess.CalledProcessError as e:
        logging.error(f"Model evaluation failed: {e.stderr}")
        raise

feature_engineering_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=run_feature_engineering,
    dag=dag,
)

model_training_task = PythonOperator(
    task_id='model_training',
    python_callable=run_model_training,
    dag=dag,
)

model_evaluation_task = PythonOperator(
    task_id='model_evaluation',
    python_callable=run_model_evaluation,
    dag=dag,
)

dvc_pipeline_task = PythonOperator(
    task_id='run_dvc_pipeline',
    python_callable=run_dvc_pipeline,
    dag=dag,
)

deploy_model_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag,
)

notification_task = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    dag=dag,
)

# Define task dependencies
check_data_task >> data_processing_task >> feature_engineering_task >> model_training_task >> model_evaluation_task >> deploy_model_task >> notification_task

# DVC pipeline task runs independently to avoid conflicts
dvc_pipeline_task