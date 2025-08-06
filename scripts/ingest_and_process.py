import pandas as pd
import numpy as np
import logging
import os
import json
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(matches_path: str, deliveries_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw IPL data."""
    logging.info(f"Loading raw matches data from {matches_path}...")
    matches_df = pd.read_csv(matches_path)

    logging.info(f"Loading raw ball-by-ball data from {deliveries_path}...")
    deliveries_df = pd.read_csv(deliveries_path)

    return matches_df, deliveries_df

def clean_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize team names across datasets."""
    # Define team name mappings
    team_mappings = {
        'Rising Pune Supergiant': 'Rising Pune Supergiants',
        'Delhi Daredevils': 'Delhi Capitals',
        'Kings XI Punjab': 'Punjab Kings',
        'Mumbai Indians': 'Mumbai Indians',
        'Royal Challengers Bangalore': 'Royal Challengers Bangalore',
        'Chennai Super Kings': 'Chennai Super Kings',
        'Kolkata Knight Riders': 'Kolkata Knight Riders',
        'Rajasthan Royals': 'Rajasthan Royals',
        'Sunrisers Hyderabad': 'Sunrisers Hyderabad',
        'Gujarat Lions': 'Gujarat Lions',
        'Pune Warriors': 'Pune Warriors',
        'Kochi Tuskers Kerala': 'Kochi Tuskers Kerala',
        'Deccan Chargers': 'Deccan Chargers',
        'Rising Pune Supergiants': 'Rising Pune Supergiants'
    }

    # Apply mappings to team columns
    for col in ['team1', 'team2', 'winner', 'batting_team', 'bowling_team']:
        if col in df.columns:
            df[col] = df[col].map(team_mappings).fillna(df[col])

    return df

def calculate_win_percentages(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate win percentages for each team."""
    # Calculate total matches and wins for each team
    team_stats = {}

    for _, match in matches_df.iterrows():
        team1, team2 = match['team1'], match['team2']
        winner = match['winner']

        # Initialize if not exists
        if team1 not in team_stats:
            team_stats[team1] = {'matches': 0, 'wins': 0}
        if team2 not in team_stats:
            team_stats[team2] = {'matches': 0, 'wins': 0}

        # Count matches
        team_stats[team1]['matches'] += 1
        team_stats[team2]['matches'] += 1

        # Count wins
        if winner == team1:
            team_stats[team1]['wins'] += 1
        elif winner == team2:
            team_stats[team2]['wins'] += 1

    # Calculate win percentages
    for team in team_stats:
        if team_stats[team]['matches'] > 0:
            team_stats[team]['win_percentage'] = team_stats[team]['wins'] / team_stats[team]['matches']
        else:
            team_stats[team]['win_percentage'] = 0.0

    return team_stats

def engineer_features(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for the dataset."""
    logging.info("Starting feature engineering...")

    # Calculate team statistics
    team_stats = calculate_win_percentages(matches_df)

    # Create features for each match
    features_list = []

    for _, match in matches_df.iterrows():
        team1, team2 = match['team1'], match['team2']

        # Get win percentages
        team1_win_pct = team_stats.get(team1, {}).get('win_percentage', 0.0)
        team2_win_pct = team_stats.get(team2, {}).get('win_percentage', 0.0)

        # Create feature row
        feature_row = {
            'id': match['id'],
            'date': match['date'],
            'Team1': team1,
            'Team2': team2,
            'City': match['city'],
            'Venue': match['venue'],
            'team1_win_percentage': team1_win_pct,
            'team2_win_percentage': team2_win_pct,
            'winner': 1 if match['winner'] == team1 else 0
        }

        features_list.append(feature_row)

    features_df = pd.DataFrame(features_list)
    logging.info("Feature engineering complete.")

    return features_df

def generate_metrics(final_df: pd.DataFrame, matches_df: pd.DataFrame, deliveries_df: pd.DataFrame):
    """Generate data quality and processing statistics metrics."""
    # Create metrics directory
    os.makedirs("metrics", exist_ok=True)

    # Data quality metrics
    data_quality = {
        "total_matches": len(matches_df),
        "total_deliveries": len(deliveries_df),
        "unique_teams": len(set(matches_df['team1'].unique()) | set(matches_df['team2'].unique())),
        "unique_venues": len(matches_df['venue'].unique()),
        "unique_cities": len(matches_df['city'].unique()),
        "missing_values": {
            "matches": matches_df.isnull().sum().to_dict(),
            "deliveries": deliveries_df.isnull().sum().to_dict()
        },
        "data_completeness": {
            "matches": (1 - matches_df.isnull().sum().sum() / (len(matches_df) * len(matches_df.columns))) * 100,
            "deliveries": (1 - deliveries_df.isnull().sum().sum() / (len(deliveries_df) * len(deliveries_df.columns))) * 100
        }
    }

    # Processing statistics
    processing_stats = {
        "processed_matches": len(final_df),
        "features_created": len(final_df.columns),
        "target_distribution": final_df['winner'].value_counts().to_dict(),
        "feature_statistics": {
            "team1_win_percentage": {
                "mean": float(final_df['team1_win_percentage'].mean()),
                "std": float(final_df['team1_win_percentage'].std()),
                "min": float(final_df['team1_win_percentage'].min()),
                "max": float(final_df['team1_win_percentage'].max())
            },
            "team2_win_percentage": {
                "mean": float(final_df['team2_win_percentage'].mean()),
                "std": float(final_df['team2_win_percentage'].std()),
                "min": float(final_df['team2_win_percentage'].min()),
                "max": float(final_df['team2_win_percentage'].max())
            }
        }
    }

    # Save metrics
    with open("metrics/data_quality.json", "w") as f:
        json.dump(data_quality, f, indent=2)

    with open("metrics/processing_stats.json", "w") as f:
        json.dump(processing_stats, f, indent=2)

    logging.info("Metrics generated and saved successfully.")

def process_ipl_data(matches_path: str, deliveries_path: str, output_path: str):
    """Main function to process IPL data."""
    try:
        # Load data
        matches_df, deliveries_df = load_data(matches_path, deliveries_path)

        # Clean data
        logging.info("Starting data cleaning and preprocessing...")
        matches_df = clean_team_names(matches_df)
        deliveries_df = clean_team_names(deliveries_df)

        # Engineer features
        final_df = engineer_features(matches_df, deliveries_df)

        # Generate metrics
        generate_metrics(final_df, matches_df, deliveries_df)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save processed data
        logging.info(f"Saving processed data to {output_path}")
        final_df.to_parquet(output_path, index=False)

        logging.info("Data processing complete and saved successfully.")

    except Exception as e:
        logging.error(f"An error occurred during data processing: {e}")
        raise

if __name__ == "__main__":
    # Define paths
    raw_matches_path = "data/raw/matches.csv"
    raw_balls_path = "data/raw/deliveries.csv"
    processed_output_path = "data/processed/processed_ipl_data.parquet"

    # Process data
    process_ipl_data(raw_matches_path, raw_balls_path, processed_output_path)
