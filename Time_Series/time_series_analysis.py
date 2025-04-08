import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import os

def perform_time_series_analysis(df, output_dir):
    results = {}
    
    # Create directory for time series plots
    ts_dir = os.path.join(output_dir, "time_series")
    if not os.path.exists(ts_dir):
        os.makedirs(ts_dir)
    
    # Plot the time series data
    plt.figure(figsize=(14, 7))
    plt.plot(df['Close'])
    plt.title('Stock Price (Close) Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.savefig(os.path.join(ts_dir, "stock_price_time_series.png"))
    plt.close()
    
    # Decompose the time series
    decomposition = seasonal_decompose(df['Close'], model='additive', period=30)  # Assuming monthly seasonality
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16))
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonality')
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residuals')
    plt.tight_layout()
    plt.savefig(os.path.join(ts_dir, "time_series_decomposition.png"))
    plt.close()
    
    # Plot ACF and PACF
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    plot_acf(df['Close'], ax=ax1)
    plot_pacf(df['Close'], ax=ax2)
    plt.savefig(os.path.join(ts_dir, "acf_pacf_plots.png"))
    plt.close()
    
    # Prepare training and testing data
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    # Fit ARIMA model
    print("\nFitting ARIMA model...")
    arima_results = fit_arima_model(train, test, ts_dir)
    results['arima'] = arima_results
    
    # Fit SARIMA model
    print("\nFitting SARIMA model...")
    sarima_results = fit_sarima_model(train, test, ts_dir)
    results['sarima'] = sarima_results
    
    # Compare models
    print("\nModel Comparison:")
    print(f"ARIMA - RMSE: {arima_results['rmse']:.4f}, MAE: {arima_results['mae']:.4f}, MAPE: {arima_results['mape']:.2f}%")
    print(f"SARIMA - RMSE: {sarima_results['rmse']:.4f}, MAE: {sarima_results['mae']:.4f}, MAPE: {sarima_results['mape']:.2f}%")
    
    # Save results summary
    with open(os.path.join(ts_dir, "time_series_results.txt"), "w") as f:
        f.write("Time Series Analysis Results\n")
        f.write("===========================\n\n")
        f.write("Dataset Information:\n")
        f.write(f"- Time Period: {df.index.min()} to {df.index.max()}\n")
        f.write(f"- Number of Observations: {len(df)}\n\n")
        
        f.write("Stationarity Test (ADF Test):\n")
        adf_result = adf_test(df['Close'])
        f.write(f"- ADF Statistic: {adf_result['adf_statistic']:.4f}\n")
        f.write(f"- p-value: {adf_result['p_value']:.4f}\n")
        f.write(f"- Is Stationary: {'Yes' if adf_result['is_stationary'] else 'No'}\n\n")
        
        f.write("ARIMA Model Results:\n")
        f.write(f"- Parameters: p={arima_results['p']}, d={arima_results['d']}, q={arima_results['q']}\n")
        f.write(f"- RMSE: {arima_results['rmse']:.4f}\n")
        f.write(f"- MAE: {arima_results['mae']:.4f}\n")
        f.write(f"- MAPE: {arima_results['mape']:.2f}%\n\n")
        
        f.write("SARIMA Model Results:\n")
        f.write(f"- Parameters: p={sarima_results['p']}, d={sarima_results['d']}, q={sarima_results['q']}, P={sarima_results['P']}, D={sarima_results['D']}, Q={sarima_results['Q']}, S={sarima_results['S']}\n")
        f.write(f"- RMSE: {sarima_results['rmse']:.4f}\n")
        f.write(f"- MAE: {sarima_results['mae']:.4f}\n")
        f.write(f"- MAPE: {sarima_results['mape']:.2f}%\n\n")
        
        f.write("Conclusion:\n")
        better_model = "ARIMA" if arima_results['rmse'] < sarima_results['rmse'] else "SARIMA"
        f.write(f"Based on RMSE, the {better_model} model performs better for this dataset.\n")
    
    return results

def adf_test(time_series):
  
    from statsmodels.tsa.stattools import adfuller
    
    result = adfuller(time_series.dropna())
    
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] <= 0.05
    }

def fit_arima_model(train, test, output_dir):
  
    # Use differenced series if it exists, otherwise use original
    target_col = 'Close_diff' if 'Close_diff' in train.columns else 'Close'
    
    # Grid search for best parameters
    print("Finding optimal ARIMA parameters...")
    best_p, best_d, best_q, best_aic = 0, 0, 0, float("inf")
    
    # Define search space - keep small for computation speed
    p_values = range(0, 3)
    d_values = range(0, 2)
    q_values = range(0, 3)
    
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(train[target_col], order=(p, d, q))
                    model_fit = model.fit()
                    current_aic = model_fit.aic
                    
                    if current_aic < best_aic:
                        best_aic = current_aic
                        best_p, best_d, best_q = p, d, q
                        
                    print(f"ARIMA({p},{d},{q}) - AIC: {current_aic:.2f}")
                except:
                    continue
    
    print(f"Best ARIMA parameters - p: {best_p}, d: {best_d}, q: {best_q}, AIC: {best_aic:.2f}")
    
    # Fit the model with best parameters
    model = ARIMA(train[target_col], order=(best_p, best_d, best_q))
    model_fit = model.fit()
    
    # Make predictions
    predictions = model_fit.forecast(steps=len(test))
    
    # If we used differenced series, convert back to original scale
    if target_col == 'Close_diff':
        # Start with the last value from training set
        last_value = train['Close'].iloc[-1]
        original_predictions = [last_value]
        
        # Integrate the differenced predictions
        for pred in predictions:
            original_predictions.append(original_predictions[-1] + pred)
        
        predictions = original_predictions[1:]  # Remove the initial value
    
    # Calculate metrics
    actual_values = test['Close'].values
    rmse = math.sqrt(mean_squared_error(actual_values, predictions))
    mae = mean_absolute_error(actual_values, predictions)
    mape = calculate_mape(actual_values, predictions)
    
    # Plot predictions vs actual
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train['Close'], label='Training Data')
    plt.plot(test.index, test['Close'], label='Actual Test Data')
    plt.plot(test.index, predictions, label=f'ARIMA({best_p},{best_d},{best_q}) Predictions', color='red')
    plt.title(f'ARIMA({best_p},{best_d},{best_q}) Model - Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "arima_predictions.png"))
    plt.close()
    
    return {
        'p': best_p,
        'd': best_d,
        'q': best_q,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'predictions': predictions
    }

def fit_sarima_model(train, test, output_dir):
   
    # Determine parameters (simplified approach)
    p, d, q = 1, 1, 1  # Non-seasonal components
    P, D, Q, S = 1, 1, 1, 5  # Seasonal components (assuming weekly seasonality)
    
    # Fit the model
    model = SARIMAX(train['Close'], order=(p, d, q), seasonal_order=(P, D, Q, S))
    model_fit = model.fit(disp=False)
    
    # Make predictions
    predictions = model_fit.forecast(steps=len(test))
    
    # Calculate metrics
    rmse = math.sqrt(mean_squared_error(test['Close'], predictions))
    mae = mean_absolute_error(test['Close'], predictions)
    mape = calculate_mape(test['Close'], predictions)
    
    # Plot predictions vs actual
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train['Close'], label='Training Data')
    plt.plot(test.index, test['Close'], label='Actual Test Data')
    plt.plot(test.index, predictions, label='SARIMA Predictions', color='red')
    plt.title(f'SARIMA({p},{d},{q})x({P},{D},{Q},{S}) Model - Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "sarima_predictions.png"))
    plt.close()
    
    return {
        'p': p,
        'd': d,
        'q': q,
        'P': P,
        'D': D,
        'Q': Q,
        'S': S,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'predictions': predictions
    }

def calculate_mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100