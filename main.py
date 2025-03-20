import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import load_and_clean_data
from src.hypothesis_testing import perform_hypothesis_testing
from src.linear_regression import perform_linear_regression

def main():
    # Define file paths
    data_path = r"D:\project\python\stat\sat-private\Data\Students_Grading_Dataset.csv"
    output_dir = r"D:\project\python\stat\sat-private\output"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set up plotting style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_clean_data(data_path)
    
    # Save the cleaned dataset
    cleaned_data_path = os.path.join(output_dir, "cleaned_data.csv")
    df.to_csv(cleaned_data_path, index=False)
    print(f"Cleaned data saved to {cleaned_data_path}")
    
    # Perform hypothesis testing
    print("\nPerforming hypothesis testing...")
    hypothesis_results = perform_hypothesis_testing(df)
    
    # Save hypothesis testing results
    with open(os.path.join(output_dir, "hypothesis_results.txt"), "w") as f:
        f.write("Hypothesis Testing Results\n")
        f.write("=========================\n\n")
        f.write("T-test for difference in Final Scores between Genders:\n")
        f.write(f"t-statistic: {hypothesis_results['t-test']['t_statistic']:.4f}\n")
        f.write(f"p-value: {hypothesis_results['t-test']['p_value']:.4f}\n\n")
        f.write("Chi-Square Test for Association between Internet Access at Home and Grade:\n")
        f.write(f"Chi-Square Statistic: {hypothesis_results['chi-square']['chi2_statistic']:.4f}\n")
        f.write(f"p-value: {hypothesis_results['chi-square']['p_value']:.4f}\n")
    
    # Perform linear regression
    print("\nPerforming linear regression analysis...")
    model, df_clean = perform_linear_regression(data_path, output_dir)
    
    print("\nAnalysis complete. Check the output directory for results.")

if __name__ == "__main__":
    main()