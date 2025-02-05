import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict

class F1DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load F1 race data from CSV file
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        return pd.read_csv(file_path)
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
        """
        Encode categorical features using LabelEncoder
        
        Args:
            df: Input DataFrame
            categorical_columns: List of categorical column names
            
        Returns:
            pd.DataFrame: DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                df_encoded[column] = self.label_encoders[column].fit_transform(df[column])
            else:
                df_encoded[column] = self.label_encoders[column].transform(df[column])
                
        return df_encoded
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        df = df.copy()
        
        # Numeric columns: fill with mean
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        # Categorical columns: fill with mode
        categorical_columns = df.select_dtypes(exclude=[np.number]).columns
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown')
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing data
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with new features
        """
        df = df.copy()
        
        # Sort by date and driver to ensure correct order for rolling calculations
        df = df.sort_values(['driverId', 'date'])
        
        # Calculate rolling average of last 5 races for each driver
        df['Last5_Avg_Position'] = df.groupby('driverId')['position'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        
        # Calculate additional features
        df['Points_per_Race'] = df.groupby('driverId')['points'].transform('mean')
        df['Grid_vs_Position'] = df['grid'] - df['position']  # Positive means improved, negative means lost positions
        df['Finish_Rate'] = df.groupby('driverId')['statusId'].transform(
            lambda x: (x == 1).rolling(window=10, min_periods=1).mean()
        )
        
        # Calculate circuit-specific performance
        df['Avg_Circuit_Position'] = df.groupby(['driverId', 'circuitId'])['position'].transform('mean')
        
        # Calculate season performance
        df['Season_Points'] = df.groupby(['driverId', 'year'])['points'].transform('cumsum')
        df['Season_Avg_Position'] = df.groupby(['driverId', 'year'])['position'].transform('mean')
        
        return df
    
    def prepare_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for model training
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target variable
        """
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Create new features
        df = self.create_features(df)
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        return X, y 