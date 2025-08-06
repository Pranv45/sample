import pandas as pd
import numpy as np
import yaml
import logging
import os
import json
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    """Load model configuration."""
    with open('configs/model_config.yaml', 'r') as file:
        return yaml.safe_load(file)

def load_features():
    """Load engineered features."""
    features_path = "data/features/engineered_features.parquet"
    df = pd.read_parquet(features_path)
    logging.info(f"Loaded features with shape: {df.shape}")
    return df

def prepare_data(df, config):
    """Prepare data for training."""
    logging.info("Preparing data for training...")

    # Select features for training
    feature_columns = []

    # Add numerical features
    for feature in config['features']['numerical_features']:
        if feature in df.columns:
            feature_columns.append(feature)

    # Add encoded categorical features
    for feature in config['features']['categorical_features']:
        encoded_col = f'{feature}_encoded'
        if encoded_col in df.columns:
            feature_columns.append(encoded_col)

    # Add interaction features
    interaction_features = ['win_percentage_diff', 'recent_form_diff', 'h2h_advantage']
    for feature in interaction_features:
        if feature in df.columns:
            feature_columns.append(feature)

    logging.info(f"Selected {len(feature_columns)} features for training")

    # Prepare X and y
    X = df[feature_columns].fillna(0)
    y = df[config['features']['target_column']]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state']
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns

def train_model(X_train, y_train, config):
    """Train the model with MLflow tracking."""
    logging.info("Starting model training with MLflow...")

    # Set up MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(config['hyperparameters'])

        # Train model
        model = xgb.XGBClassifier(**config['hyperparameters'])
        model.fit(X_train, y_train)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Log feature importance
        feature_importance = dict(zip(range(len(model.feature_importances_)), model.feature_importances_))
        mlflow.log_dict(feature_importance, "feature_importance.json")

        logging.info("Model training completed successfully")
        return model

def evaluate_model(model, X_test, y_test, config):
    """Evaluate the model and log metrics."""
    logging.info("Evaluating model...")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", classification_rep['weighted avg']['precision'])
    mlflow.log_metric("recall", classification_rep['weighted avg']['recall'])
    mlflow.log_metric("f1_score", classification_rep['weighted avg']['f1-score'])

    # Save metrics to file
    metrics = {
        'accuracy': accuracy,
        'precision': classification_rep['weighted avg']['precision'],
        'recall': classification_rep['weighted avg']['recall'],
        'f1_score': classification_rep['weighted avg']['f1-score'],
        'classification_report': classification_rep,
        'confusion_matrix': conf_matrix.tolist(),
        'timestamp': datetime.now().isoformat()
    }

    os.makedirs('metrics', exist_ok=True)
    with open('metrics/model_performance.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    logging.info(f"Model evaluation completed. Accuracy: {accuracy:.4f}")
    return metrics

def save_model(model, scaler, feature_columns, config):
    """Save the trained model and artifacts."""
    logging.info("Saving model and artifacts...")

    os.makedirs('models', exist_ok=True)

    # Save model
    model_path = f"models/{config['model']['name']}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save scaler
    scaler_path = "models/scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Save feature columns
    feature_info = {
        'feature_columns': feature_columns,
        'model_name': config['model']['name'],
        'model_type': config['model']['type'],
        'training_date': datetime.now().isoformat()
    }

    with open('models/model_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)

    logging.info(f"Model saved to {model_path}")

def main():
    """Main training pipeline."""
    try:
        # Load configuration
        config = load_config()

        # Load features
        df = load_features()

        # Prepare data
        X_train, X_test, y_train, y_test, scaler, feature_columns = prepare_data(df, config)

        # Train model
        model = train_model(X_train, y_train, config)

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, config)

        # Save model
        save_model(model, scaler, feature_columns, config)

        logging.info("Training pipeline completed successfully!")

    except Exception as e:
        logging.error(f"Error in training pipeline: {e}")
        raise

if __name__ == "__main__":
    main()