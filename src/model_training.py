import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Tuple
import xgboost as xgb
import lightgbm as lgb

class F1ModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.best_score = float('inf')  # Lower is better for regression metrics
        
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """
        Split data into training and testing sets
        
        Args:
            X: Features DataFrame
            y: Target variable
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, random_state=42)
    
    def evaluate_model(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dict[str, float]: Dictionary containing various performance metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
        # Calculate basic regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate additional metrics
        explained_variance = 1 - np.var(y_test - y_pred) / np.var(y_test)
        
        # Calculate position-specific metrics
        position_accuracy = np.mean(np.abs(np.round(y_pred) - y_test) <= 1)  # Within 1 position
        top_3_accuracy = np.mean((y_test <= 3) == (y_pred <= 3))  # Podium prediction accuracy
        points_accuracy = np.mean((y_test <= 10) == (y_pred <= 10))  # Points prediction accuracy
        
        # Calculate error distribution
        errors = y_pred - y_test
        error_std = np.std(errors)
        error_percentiles = np.percentile(errors, [25, 50, 75])
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'explained_variance': explained_variance,
            'position_accuracy': position_accuracy,
            'top_3_accuracy': top_3_accuracy,
            'points_accuracy': points_accuracy,
            'error_std': error_std,
            'error_25th': error_percentiles[0],
            'error_median': error_percentiles[1],
            'error_75th': error_percentiles[2]
        }
        
        return metrics
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, model_name: str):
        """
        Train a specific model
        
        Args:
            X_train: Training features
            y_train: Training target
            model_name: Name of the model to train
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        print(f"\nTraining {model_name.replace('_', ' ').title()}...")
        self.models[model_name].fit(X_train, y_train)
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.models[model_name], X_train, y_train, 
                                  cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        print(f"Cross-validation RMSE: {cv_rmse.mean():.3f} (+/- {cv_rmse.std() * 2:.3f})")
    
    def train_and_evaluate_all(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                             y_train: pd.Series, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Train and evaluate all models
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            
        Returns:
            Dict[str, Dict[str, float]]: Dictionary containing performance metrics for all models
        """
        results = {}
        
        for model_name in self.models:
            # Train model
            self.train_model(X_train, y_train, model_name)
            
            # Evaluate model
            metrics = self.evaluate_model(model_name, X_test, y_test)
            results[model_name] = metrics
            
            # Track best model (using RMSE as the criterion)
            if metrics['rmse'] < self.best_score:
                self.best_score = metrics['rmse']
                self.best_model = model_name
                
            # Print detailed metrics
            print(f"\n{model_name.replace('_', ' ').title()} Performance:")
            print(f"MSE: {metrics['mse']:.3f}")
            print(f"RMSE: {metrics['rmse']:.3f}")
            print(f"MAE: {metrics['mae']:.3f}")
            print(f"R²: {metrics['r2']:.3f}")
            print(f"Explained Variance: {metrics['explained_variance']:.3f}")
            print(f"Position Accuracy (±1): {metrics['position_accuracy']:.1%}")
            print(f"Podium Prediction Accuracy: {metrics['top_3_accuracy']:.1%}")
            print(f"Points Prediction Accuracy: {metrics['points_accuracy']:.1%}")
            print("\nError Distribution:")
            print(f"Standard Deviation: {metrics['error_std']:.3f}")
            print(f"25th Percentile: {metrics['error_25th']:.3f}")
            print(f"Median Error: {metrics['error_median']:.3f}")
            print(f"75th Percentile: {metrics['error_75th']:.3f}")
        
        print(f"\nBest Model: {self.best_model.replace('_', ' ').title()} (RMSE: {self.best_score:.3f})")
        return results
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """
        Make predictions using a specific model or the best model
        
        Args:
            X: Features to predict on
            model_name: Name of the model to use (if None, uses best model)
            
        Returns:
            np.ndarray: Predictions
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No model has been trained yet")
            model_name = self.best_model
            
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        return self.models[model_name].predict(X) 