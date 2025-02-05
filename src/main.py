import pandas as pd
import numpy as np
from data_processing import F1DataProcessor
from model_training import F1ModelTrainer
from visualization import F1Visualizer
import logging
from typing import Dict, Any
import os
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_process_data() -> tuple:
    """
    Load and process F1 race data from archive folder
    
    Returns:
        tuple: Processed features and target variables
    """
    logger.info("Loading and processing data...")
    processor = F1DataProcessor()
    
    # Load data from archive folder
    data_path = os.path.join('archive', 'races.csv')
    circuits_path = os.path.join('archive', 'circuits.csv')
    results_path = os.path.join('archive', 'results.csv')
    drivers_path = os.path.join('archive', 'drivers.csv')
    constructors_path = os.path.join('archive', 'constructors.csv')
    
    # Load all datasets
    races_df = pd.read_csv(data_path)
    circuits_df = pd.read_csv(circuits_path)
    results_df = pd.read_csv(results_path)
    drivers_df = pd.read_csv(drivers_path)
    constructors_df = pd.read_csv(constructors_path)
    
    # Select only necessary columns before merging
    races_df = races_df[['raceId', 'year', 'round', 'circuitId', 'name', 'date']]
    circuits_df = circuits_df[['circuitId', 'circuitRef', 'name', 'location', 'country', 'lat', 'lng']]
    results_df = results_df[['resultId', 'raceId', 'driverId', 'constructorId', 'grid', 'position', 'points', 'laps', 'statusId']]
    drivers_df = drivers_df[['driverId', 'driverRef', 'number', 'code', 'forename', 'surname', 'nationality']]
    constructors_df = constructors_df[['constructorId', 'constructorRef', 'name', 'nationality']]
    
    # Rename columns to avoid conflicts
    races_df = races_df.rename(columns={'name': 'race_name'})
    circuits_df = circuits_df.rename(columns={'name': 'circuit_name'})
    constructors_df = constructors_df.rename(columns={'name': 'constructor_name', 'nationality': 'constructor_nationality'})
    
    # Merge datasets
    df = results_df.merge(races_df, on='raceId')
    df = df.merge(drivers_df, on='driverId')
    df = df.merge(constructors_df, on='constructorId')
    df = df.merge(circuits_df, on='circuitId')
    
    # Create useful features
    df['full_name'] = df['forename'] + ' ' + df['surname']
    df['circuit_location'] = df['location'] + ', ' + df['country']
    
    # Convert position to numeric, handling any non-numeric values
    df['position'] = pd.to_numeric(df['position'], errors='coerce')
    
    # Handle missing values
    df = processor.handle_missing_values(df)
    
    # Create features
    df = processor.create_features(df)
    
    # Store original data before encoding
    df_original = df.copy()
    
    # Define categorical columns
    categorical_columns = ['full_name', 'constructorRef', 'circuitRef', 'driverRef']
    df = processor.encode_categorical_features(df, categorical_columns)
    
    # Select features for modeling
    feature_columns = [
        # Basic race information
        'grid', 'year', 'round', 'laps',
        
        # Encoded categorical features
        'full_name', 'constructorRef', 'circuitRef', 'driverRef',
        
        # Performance metrics
        'Last5_Avg_Position',
        'Points_per_Race',
        'Grid_vs_Position',
        'Finish_Rate',
        'Avg_Circuit_Position',
        'Season_Points',
        'Season_Avg_Position'
    ]
    
    # Prepare data for modeling
    X = df[feature_columns]
    y = df['position']
    
    return X, y, df_original  # Return original DataFrame for visualization

def train_and_evaluate_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Train and evaluate multiple models
    
    Args:
        X: Feature DataFrame
        y: Target variable
        
    Returns:
        Dict[str, Any]: Dictionary containing trained models and their performance metrics
    """
    logger.info("Training and evaluating models...")
    trainer = F1ModelTrainer()
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    # Train and evaluate all models
    results = trainer.train_and_evaluate_all(X_train, X_test, y_train, y_test)
    
    return {
        'trainer': trainer,
        'results': results,
        'test_data': (X_test, y_test)
    }

def visualize_results(df: pd.DataFrame, model_results: Dict[str, Any], feature_names: list):
    """
    Create visualizations of the results
    
    Args:
        df: Original DataFrame (non-encoded)
        model_results: Dictionary containing model results
        feature_names: List of feature names
    """
    logger.info("Creating visualizations...")
    visualizer = F1Visualizer()
    
    # Plot feature importance for the best model
    trainer = model_results['trainer']
    best_model = trainer.models[trainer.best_model]
    visualizer.plot_feature_importance(best_model, feature_names)
    
    # Plot predictions vs actual for test data
    X_test, y_test = model_results['test_data']
    y_pred = trainer.predict(X_test)
    visualizer.plot_prediction_vs_actual(y_test, y_pred)
    
    # Plot correlation matrix for numeric features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    visualizer.plot_correlation_matrix(df, numeric_features)
    
    # Get top drivers and teams
    top_drivers = df.groupby('full_name')['points'].sum().sort_values(ascending=False).head(5)
    print("\nTop 5 Drivers by Total Points:")
    print(top_drivers)
    
    for driver in top_drivers.index:
        visualizer.plot_driver_performance(df, driver, 'points')
        visualizer.plot_driver_performance(df, driver, 'position')
    
    top_teams = df.groupby('constructor_name')['points'].sum().sort_values(ascending=False).head(5)
    print("\nTop 5 Constructors by Total Points:")
    print(top_teams)
    
    visualizer.plot_team_comparison(df, list(top_teams.index), 'points')
    visualizer.plot_team_comparison(df, list(top_teams.index), 'position')
    
    # Analyze popular circuits
    popular_circuits = df['circuit_name'].value_counts().head(3)
    print("\nMost Popular Circuits:")
    print(popular_circuits)
    
    for circuit in popular_circuits.index:
        visualizer.plot_circuit_performance(df, circuit)

def main():
    """
    Main function to run the F1 prediction system
    """
    try:
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        os.makedirs('results/images', exist_ok=True)
        
        # Configure logging to save to file
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('results/training_log.txt'),
                logging.StreamHandler()
            ]
        )
        
        # Load and process data
        X, y, df_original = load_and_process_data()
        
        # Train and evaluate models
        model_results = train_and_evaluate_models(X, y)
        
        # Save model performance results
        results_df = pd.DataFrame()
        for model_name, metrics in model_results['results'].items():
            metrics_df = pd.DataFrame([metrics])
            metrics_df['model'] = model_name
            results_df = pd.concat([results_df, metrics_df], ignore_index=True)
        
        results_df.to_csv('results/model_performance.csv', index=False)
        
        # Create visualizations
        feature_names = X.columns.tolist()
        visualizer = F1Visualizer()
        
        # Save current matplotlib backend
        current_backend = plt.get_backend()
        # Switch to Agg backend for saving figures
        plt.switch_backend('Agg')
        
        # Plot and save feature importance
        trainer = model_results['trainer']
        best_model = trainer.models[trainer.best_model]
        plt.figure(figsize=(12, 8))
        visualizer.plot_feature_importance(best_model, feature_names)
        plt.savefig('results/images/feature_importance.png')
        plt.close()
        
        # Plot and save predictions vs actual
        X_test, y_test = model_results['test_data']
        y_pred = trainer.predict(X_test)
        plt.figure(figsize=(12, 8))
        visualizer.plot_prediction_vs_actual(y_test, y_pred)
        plt.savefig('results/images/prediction_vs_actual.png')
        plt.close()
        
        # Get top drivers and teams
        top_drivers = df_original.groupby('full_name')['points'].sum().sort_values(ascending=False).head(5)
        
        # Save driver performance plots
        for driver in top_drivers.index:
            plt.figure(figsize=(12, 8))
            visualizer.plot_driver_performance(df_original, driver, 'points')
            plt.savefig(f'results/images/{driver.replace(" ", "_")}_performance.png')
            plt.close()
            
            plt.figure(figsize=(12, 8))
            visualizer.plot_driver_performance(df_original, driver, 'position')
            plt.savefig(f'results/images/{driver.replace(" ", "_")}_positions.png')
            plt.close()
        
        # Save team comparison plots
        top_teams = df_original.groupby('constructor_name')['points'].sum().sort_values(ascending=False).head(5)
        plt.figure(figsize=(12, 8))
        visualizer.plot_team_comparison(df_original, list(top_teams.index), 'points')
        plt.savefig('results/images/team_comparison_points.png')
        plt.close()
        
        plt.figure(figsize=(12, 8))
        visualizer.plot_team_comparison(df_original, list(top_teams.index), 'position')
        plt.savefig('results/images/team_comparison_positions.png')
        plt.close()
        
        # Save correlation matrix
        numeric_features = df_original.select_dtypes(include=[np.number]).columns.tolist()
        plt.figure(figsize=(12, 10))
        visualizer.plot_correlation_matrix(df_original, numeric_features)
        plt.savefig('results/images/correlation_matrix.png')
        plt.close()
        
        # Switch back to original backend
        plt.switch_backend(current_backend)
        
        # Generate summary report
        with open('results/summary.txt', 'w') as f:
            f.write("F1 Race Prediction System - Results Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Model Performance:\n")
            f.write("-" * 20 + "\n")
            for model_name, metrics in model_results['results'].items():
                f.write(f"\n{model_name}:\n")
                for metric_name, value in metrics.items():
                    f.write(f"{metric_name}: {value:.4f}\n")
            
            f.write("\nTop 5 Drivers by Total Points:\n")
            f.write("-" * 20 + "\n")
            f.write(top_drivers.to_string())
            
            f.write("\n\nTop 5 Constructors by Total Points:\n")
            f.write("-" * 20 + "\n")
            f.write(top_teams.to_string())
        
        logger.info("Results have been saved to the 'results' directory")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 