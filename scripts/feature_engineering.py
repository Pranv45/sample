import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import yaml
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    """Load model configuration."""
    with open('configs/model_config.yaml', 'r') as file:
        return yaml.safe_load(file)

def engineer_features(df):
    """Engineer advanced features for the IPL win prediction model."""
    config = load_config()

    logging.info("Starting advanced feature engineering...")

    # Convert to pandas for easier feature engineering
    df_pandas = df.toPandas()

    # 1. Recent Form (last 5 matches)
    logging.info("Calculating recent form features...")
    df_pandas['team1_recent_form'] = df_pandas.groupby('Team1')['winner'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    df_pandas['team2_recent_form'] = df_pandas.groupby('Team2')['winner'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)

    # 2. Head-to-Head Performance
    logging.info("Calculating head-to-head features...")
    h2h_stats = df_pandas.groupby(['Team1', 'Team2']).agg({
        'winner': ['count', 'sum']
    }).reset_index()
    h2h_stats.columns = ['Team1', 'Team2', 'h2h_matches', 'h2h_wins_team1']
    h2h_stats['h2h_win_rate_team1'] = h2h_stats['h2h_wins_team1'] / h2h_stats['h2h_matches']

    # Merge head-to-head stats
    df_pandas = df_pandas.merge(h2h_stats, on=['Team1', 'Team2'], how='left')

    # Create team2 head-to-head stats
    h2h_stats_team2 = h2h_stats.copy()
    h2h_stats_team2['Team1'], h2h_stats_team2['Team2'] = h2h_stats_team2['Team2'], h2h_stats_team2['Team1']
    h2h_stats_team2['h2h_win_rate_team2'] = 1 - h2h_stats_team2['h2h_win_rate_team1']

    df_pandas = df_pandas.merge(h2h_stats_team2[['Team1', 'Team2', 'h2h_win_rate_team2']],
                                on=['Team1', 'Team2'], how='left')

    # 3. Venue Performance
    logging.info("Calculating venue performance features...")
    venue_stats = df_pandas.groupby(['Team1', 'Venue']).agg({
        'winner': ['count', 'sum']
    }).reset_index()
    venue_stats.columns = ['Team1', 'Venue', 'venue_matches_team1', 'venue_wins_team1']
    venue_stats['venue_win_rate_team1'] = venue_stats['venue_wins_team1'] / venue_stats['venue_matches_team1']

    df_pandas = df_pandas.merge(venue_stats, on=['Team1', 'Venue'], how='left')

    # 4. Season Performance
    logging.info("Calculating season performance features...")
    # Use 'date' instead of 'Date' for season extraction
    df_pandas['Season'] = pd.to_datetime(df_pandas['date']).dt.year
    season_stats = df_pandas.groupby(['Team1', 'Season']).agg({
        'winner': ['count', 'sum']
    }).reset_index()
    season_stats.columns = ['Team1', 'Season', 'season_matches_team1', 'season_wins_team1']
    season_stats['season_win_rate_team1'] = season_stats['season_wins_team1'] / season_stats['season_matches_team1']

    df_pandas = df_pandas.merge(season_stats, on=['Team1', 'Season'], how='left')

    # 5. Categorical Encoding
    logging.info("Encoding categorical features...")
    label_encoders = {}
    categorical_features = config['features']['categorical_features']

    for feature in categorical_features:
        if feature in df_pandas.columns:
            le = LabelEncoder()
            df_pandas[f'{feature}_encoded'] = le.fit_transform(df_pandas[feature].astype(str))
            label_encoders[feature] = le

    # 6. Fill missing values
    logging.info("Filling missing values...")
    numerical_features = config['features']['numerical_features']
    for feature in numerical_features:
        if feature in df_pandas.columns:
            df_pandas[feature] = df_pandas[feature].fillna(df_pandas[feature].median())

    # 7. Create interaction features
    logging.info("Creating interaction features...")
    df_pandas['win_percentage_diff'] = df_pandas['team1_win_percentage'] - df_pandas['team2_win_percentage']
    df_pandas['recent_form_diff'] = df_pandas['team1_recent_form'] - df_pandas['team2_recent_form']
    df_pandas['h2h_advantage'] = df_pandas['h2h_win_rate_team1'] - df_pandas['h2h_win_rate_team2']

    # Save feature importance metrics
    feature_importance = {
        'features_created': list(df_pandas.columns),
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'total_features': len(df_pandas.columns),
        'timestamp': datetime.now().isoformat()
    }

    # Save feature importance metrics
    os.makedirs('metrics', exist_ok=True)
    import json
    with open('metrics/feature_importance.json', 'w') as f:
        json.dump(feature_importance, f, indent=2)

    logging.info(f"Feature engineering complete. Created {len(df_pandas.columns)} features.")

    return df_pandas, label_encoders

def save_features(df, output_dir):
    """Save engineered features."""
    os.makedirs(output_dir, exist_ok=True)

    # Save as parquet
    df.to_parquet(f'{output_dir}/engineered_features.parquet', index=False)

    # Save label encoders
    import pickle
    with open(f'{output_dir}/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)

    logging.info(f"Features saved to {output_dir}")

if __name__ == "__main__":
    # Load processed data
    import pyspark
    from pyspark.sql import SparkSession

    spark = SparkSession.builder \
        .appName("FeatureEngineering") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    # Load processed data
    processed_data_path = "data/processed/processed_ipl_data.parquet"
    df = spark.read.parquet(processed_data_path)

    # Engineer features
    df_engineered, label_encoders = engineer_features(df)

    # Save features
    save_features(df_engineered, "data/features")

    spark.stop()