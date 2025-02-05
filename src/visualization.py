import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict

class F1Visualizer:
    def __init__(self):
        # Set style for all plots
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['font.size'] = 10
        
        # Fix Turkish character encoding
        plt.rcParams['font.family'] = 'DejaVu Sans'
        
    def plot_driver_performance(self, df: pd.DataFrame, driver: str, metric: str = 'position'):
        """
        Plot performance metrics for a specific driver over time
        
        Args:
            df: DataFrame containing race data
            driver: Name of the driver
            metric: Performance metric to plot
        """
        plt.figure(figsize=(14, 7))
        driver_data = df[df['full_name'] == driver].copy()
        driver_data = driver_data.sort_values('date')
        
        # Calculate moving average
        window = 5
        driver_data[f'{metric}_ma'] = driver_data[metric].rolling(window=window, min_periods=1).mean()
        
        # Plot actual values and moving average
        sns.scatterplot(data=driver_data, x='year', y=metric, alpha=0.5, label='Actual')
        sns.lineplot(data=driver_data, x='year', y=f'{metric}_ma', color='red', 
                    label=f'{window}-Race Moving Average')
        
        metric_name = metric.replace('_', ' ').title()
        plt.title(f"{driver} - {metric_name} Performance Over Time")
        plt.xlabel('Year')
        plt.ylabel(metric_name)
        plt.xticks(rotation=45)
        plt.legend()
        
        # Add statistics annotation
        stats = (f"Average: {driver_data[metric].mean():.2f}\n"
                f"Best: {driver_data[metric].min():.2f}\n"
                f"Worst: {driver_data[metric].max():.2f}")
        plt.annotate(stats, xy=(0.02, 0.95), xycoords='axes fraction',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        plt.tight_layout()
        plt.show()
        
    def plot_team_comparison(self, df: pd.DataFrame, teams: List[str], metric: str = 'points'):
        """
        Compare performance metrics between teams
        
        Args:
            df: DataFrame containing race data
            teams: List of team names to compare
            metric: Performance metric to compare
        """
        plt.figure(figsize=(14, 7))
        
        team_data = df[df['constructor_name'].isin(teams)].copy()
        
        # Create box plot instead of violin plot
        sns.boxplot(data=team_data, x='constructor_name', y=metric, 
                   showfliers=True, # Show outlier points
                   width=0.7,       # Width of the boxes
                   whis=1.5)        # Length of the whiskers
        
        # Add individual points with smaller size and more transparency
        sns.stripplot(data=team_data, x='constructor_name', y=metric,
                     color='black', alpha=0.2, size=2, jitter=0.2)
        
        metric_name = metric.replace('_', ' ').title()
        plt.title(f'Constructor Comparison - {metric_name}')
        plt.xlabel('Constructor')
        plt.ylabel(metric_name)
        plt.xticks(rotation=45, ha='right')
        
        # Calculate and display statistics
        team_stats = team_data.groupby('constructor_name')[metric].agg([
            ('Races', 'count'),
            ('Mean', 'mean'),
            ('Median', 'median'),
            ('Std', 'std'),
            ('Min', 'min'),
            ('Max', 'max')
        ]).round(2)
        
        print(f"\nTeam Statistics for {metric_name}:")
        print(team_stats)
        
        # Add statistics annotation to the plot
        for i, team in enumerate(teams):
            stats = team_stats.loc[team]
            text = f"Races: {stats['Races']}\nMean: {stats['Mean']:.1f}\nMedian: {stats['Median']:.1f}"
            plt.annotate(text, xy=(i, team_data[metric].min()), 
                        xytext=(0, -20), textcoords='offset points',
                        ha='center', va='top',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        plt.tight_layout()
        plt.show()
        
    def plot_circuit_performance(self, df: pd.DataFrame, circuit: str, n_top_drivers: int = 5):
        """
        Analyze and plot driver performance at a specific circuit
        
        Args:
            df: DataFrame containing race data
            circuit: Name of the circuit
            n_top_drivers: Number of top drivers to show
        """
        circuit_data = df[df['circuit_name'] == circuit].copy()
        
        # Calculate comprehensive statistics
        driver_stats = circuit_data.groupby('full_name').agg({
            'position': ['count', 'mean', 'std', 'min', 'max'],
            'points': ['sum', 'mean']
        }).round(2)
        
        # Flatten column names
        driver_stats.columns = ['Races', 'Avg Pos', 'Std Pos', 'Best Pos', 'Worst Pos', 'Total Points', 'Avg Points']
        driver_stats = driver_stats[driver_stats['Races'] > 3]  # Filter drivers with more than 3 races
        driver_stats = driver_stats.sort_values('Avg Pos')
        
        print(f"\nTop Drivers at {circuit}:")
        print(driver_stats.head(n_top_drivers))
        
        # Create visualization
        plt.figure(figsize=(14, 7))
        top_drivers = driver_stats.head(n_top_drivers)
        
        # Plot average positions with error bars
        plt.errorbar(
            x=range(len(top_drivers)),
            y=top_drivers['Avg Pos'],
            yerr=top_drivers['Std Pos'],
            fmt='o',
            capsize=5,
            label='Average Position ± Std Dev'
        )
        
        # Add best and worst positions
        plt.plot(range(len(top_drivers)), top_drivers['Best Pos'], 'g^', label='Best Position')
        plt.plot(range(len(top_drivers)), top_drivers['Worst Pos'], 'rv', label='Worst Position')
        
        plt.title(f'Driver Performance at {circuit}')
        plt.xlabel('Driver')
        plt.ylabel('Position (lower is better)')
        plt.xticks(range(len(top_drivers)), top_drivers.index, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def plot_correlation_matrix(self, df: pd.DataFrame, features: List[str]):
        """
        Plot correlation matrix for selected features
        
        Args:
            df: DataFrame containing race data
            features: List of features to include in correlation matrix
        """
        # Select only numeric features and rename them for better readability
        numeric_features = df[features].select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_features].corr()
        
        # Rename features for better readability
        feature_names = [f.replace('_', ' ').title() for f in numeric_features]
        correlation_matrix.index = feature_names
        correlation_matrix.columns = feature_names
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix), k=1)
        
        # Create heatmap with improved aesthetics
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            mask=mask,
            square=True,
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
    def plot_feature_importance(self, model, feature_names: List[str]):
        """
        Plot feature importance from a trained model
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature importances")
            
        # Create feature importance DataFrame with readable names
        importances = pd.DataFrame({
            'feature': [f.replace('_', ' ').title() for f in feature_names],
            'importance': model.feature_importances_
        })
        importances = importances.sort_values('importance', ascending=True)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importances, x='importance', y='feature')
        plt.title('Feature Importance in Model Predictions')
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.show()
        
        # Print feature importances with percentages
        print("\nFeature Importance Rankings:")
        total_importance = importances['importance'].sum()
        for idx, row in importances.iterrows():
            percentage = (row['importance'] / total_importance) * 100
            print(f"{row['feature']}: {percentage:.1f}%")
        
    def plot_prediction_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, title: str = 'Predictions vs Actual'):
        """
        Plot predicted vs actual values
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
        """
        plt.figure(figsize=(14, 7))
        
        # Create scatter plot with density
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Position')
        plt.ylabel('Predicted Position')
        plt.title(title)
        plt.legend()
        
        # Calculate and display error metrics
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        
        stats = (f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.3f}")
        plt.annotate(stats, xy=(0.02, 0.95), xycoords='axes fraction',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        plt.tight_layout()
        plt.show()
        
        # Plot error distribution
        plt.figure(figsize=(14, 7))
        residuals = y_pred - y_true
        
        sns.histplot(residuals, kde=True, bins=30)
        plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
        plt.title('Prediction Error Distribution')
        plt.xlabel('Error (Predicted - Actual)')
        plt.ylabel('Count')
        plt.legend()
        
        # Add error statistics
        stats = (f"Mean Error: {np.mean(residuals):.2f}\n"
                f"Std Error: {np.std(residuals):.2f}\n"
                f"Median Error: {np.median(residuals):.2f}\n"
                f"90% Error Range: [{np.percentile(residuals, 5):.2f}, {np.percentile(residuals, 95):.2f}]")
        plt.annotate(stats, xy=(0.02, 0.95), xycoords='axes fraction',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        plt.tight_layout()
        plt.show() 