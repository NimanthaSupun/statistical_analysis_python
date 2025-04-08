
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def load_and_clean_data(file_path, is_time_series=False):
  
    df = pd.read_csv(file_path)
    
    # Display basic information
    print("Dataset Info:")
    print(df.info())
    print("\nDataset Description:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    if is_time_series:
        return process_time_series_data(df)
    else:
        return process_regular_data(df)

def process_regular_data(df):
   
    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])  # Fill categorical columns with mode
            else:
                df[col] = df[col].fillna(df[col].median())  # Fill numerical columns with median
    
    print("\nMissing values handled.")
    print(df.isnull().sum())
        
    return df

def process_time_series_data(df):
    
    df = df.copy()
    
    # Convert date to datetime with UTC=True to handle timezone warnings
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        
        # Remove duplicate dates before setting the index
        if df['Date'].duplicated().any():
            print("\nRemoving duplicate dates from the 'Date' column.")
            df = df[~df['Date'].duplicated(keep='first')]
        
        df.set_index('Date', inplace=True)
        
        # Ensure the index has a specific frequency
        # For daily data use 'D', for business days 'B', etc.
        df = df.asfreq('D', method='pad')  # Forward fill any new missing values from reindexing
    
    # Remove duplicate dates if any (already handled above)
    if df.index.duplicated().any():
        print("\nRemoving duplicate dates from the index.")
        df = df[~df.index.duplicated(keep='first')]
    
    # Sort by date
    df = df.sort_index()
    
    # Drop unnecessary columns for time series analysis
    columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
    columns_to_drop = [col for col in df.columns if col not in columns_to_keep]
    if columns_to_drop:
        print(f"\nDropping columns: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
    
    # Handle missing values in time series
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            # For time series, use forward fill followed by backward fill
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    print("\nMissing values handled.")
    
    # Check for stationarity
    print("\nChecking stationarity of 'Close' prices:")
    is_stationary = check_stationarity(df['Close'])
    
    # If not stationary, create differenced series
    if not is_stationary:
        print("Creating differenced series for modeling...")
        df['Close_diff'] = df['Close'].diff().dropna()
        # Check stationarity of differenced series
        print("\nChecking stationarity of differenced 'Close' prices:")
        check_stationarity(df['Close_diff'])
    
    return df

def check_stationarity(time_series):

    result = adfuller(time_series.dropna())
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    
    is_stationary = result[1] <= 0.05
    if is_stationary:
        print("Time series is stationary")
    else:
        print("Time series is not stationary")
    
    return is_stationary
