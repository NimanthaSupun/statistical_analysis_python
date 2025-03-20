import pandas as pd
import numpy as np

def load_and_clean_data(file_path):
    """Load dataset, check data types, handle missing values, and return cleaned DataFrame."""
    df = pd.read_csv(file_path)
    
    # Display basic information
    print("Dataset Info:")
    print(df.info())
    print("\nDataset Description:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])  # Fill categorical columns with mode
            else:
                df[col] = df[col].fillna(df[col].median())  # Fill numerical columns with median
    
    print("\nMissing values handled.")
    
    return df
