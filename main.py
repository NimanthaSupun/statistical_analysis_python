import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Data_Cleaning.preprocessing import load_and_clean_data
from Regression.linear_regression import perform_linear_regression
from Hypothesis.hypothesis_testing import perform_hypothesis_testing
from Time_Series.time_series_analysis import perform_time_series_analysis

def main():
    # Define file paths
    student_data_path = r"D:\project\python\stat\sat-private\Datasets\Students_Grading_Dataset.csv"
    time_series_data_path = r"D:\project\python\stat\sat-private\Datasets\apple_only.csv"
    output_dir = r"D:\project\python\stat\sat-private\output"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set up plotting style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # PART 1: Student Grading Dataset Analysis
    print("=" * 80)
    print("PART 1: STUDENT GRADING DATASET ANALYSIS")
    print("=" * 80)
    
    # Load and preprocess data
    print("Loading and preprocessing student data...")
    student_df = load_and_clean_data(student_data_path, is_time_series=False)
    
    # Save the cleaned dataset
    cleaned_data_path = os.path.join(output_dir, "cleaned_student_data.csv")
    student_df.to_csv(cleaned_data_path, index=False)
    print(f"Cleaned student data saved to {cleaned_data_path}")
    
    # Perform hypothesis testing
    print("\nPerforming hypothesis testing...")
    hypothesis_results = perform_hypothesis_testing(student_df,output_dir=output_dir)
 
    # Perform linear regression
    print("\nPerforming linear regression analysis...")
    model, df_clean = perform_linear_regression(student_data_path, output_dir)
    
    # PART 2: Time Series Analysis
    print("\n" + "=" * 80)
    print("PART 2: TIME SERIES ANALYSIS")
    print("=" * 80)
    
    # Load and preprocess time series data
    print("Loading and preprocessing time series data...")
    ts_df = load_and_clean_data(time_series_data_path, is_time_series=True)
    
    # Save the cleaned time series dataset
    cleaned_ts_path = os.path.join(output_dir, "cleaned_time_series_data.csv")
    ts_df.to_csv(cleaned_ts_path)
    print(f"Cleaned time series data saved to {cleaned_ts_path}")
    
    # Perform time series analysis
    print("\nPerforming time series analysis...")
    ts_results = perform_time_series_analysis(ts_df, output_dir)
    
    print("\nAnalysis complete. Check the output directory for results.")

if __name__ == "__main__":
    main()
    