import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_linear_regression(data_path, output_dir="output"):
    """
    Perform linear regression analysis on student data.
    
    Args:
        data_path (str): Path to the CSV file
        output_dir (str): Directory to save output files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load dataset
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Display basic information about the dataset
    print(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Check for missing values in key columns
    predictors = ['Midterm_Score', 'Assignments_Avg', 'Quizzes_Avg', 'Participation_Score']
    missing_counts = df[predictors + ['Final_Score']].isnull().sum()
    print("\nMissing values in key columns:")
    print(missing_counts)
    
    # Handle missing values
    print("\nHandling missing values...")
    # Option 1: Drop rows with missing values in key columns
    df_clean = df.dropna(subset=['Final_Score'] + predictors)
    print(f"Rows after dropping missing values: {df_clean.shape[0]}")
    
    # Or Option 2: Impute missing values
    # df_clean = df.copy()
    # for col in predictors:
    #     df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Select dependent and independent variables
    y = df_clean['Final_Score']
    X = df_clean[predictors]
    
    # Add constant for intercept
    X = sm.add_constant(X)
    
    # Fit regression model
    print("\nFitting regression model...")
    model = sm.OLS(y, X).fit()
    
    # Print regression summary
    print("\nRegression Summary:")
    print(model.summary().as_text())
    
    # Save regression summary to file
    with open(os.path.join(output_dir, "regression_summary.txt"), "w") as f:
        f.write(model.summary().as_text())
    
    # Check for multicollinearity
    print("\nChecking for multicollinearity...")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("Variance Inflation Factor (VIF):")
    print(vif_data)
    
    # Save VIF table to file
    vif_data.to_csv(os.path.join(output_dir, "vif_table.csv"), index=False)
    
    # Run ANOVA using formula API
    print("\nRunning ANOVA...")
    formula = 'Final_Score ~ Midterm_Score + Assignments_Avg + Quizzes_Avg + Participation_Score'
    try:
        # Use formula API for ANOVA
        ols_model = smf.ols(formula, data=df_clean).fit()
        anova_table = sm.stats.anova_lm(ols_model, typ=2)
        print("ANOVA Table:")
        print(anova_table)
        
        # Save ANOVA table to file
        anova_table.to_csv(os.path.join(output_dir, "anova_table.csv"))
    except Exception as e:
        print(f"ANOVA could not be performed: {str(e)}")
    
    # Create diagnostic plots
    print("\nCreating diagnostic plots...")
    
    # Residuals vs Fitted Values
    plt.figure(figsize=(10, 6))
    sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, 
                  line_kws={"color": "red", "lw": 1})
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted Values")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residuals_plot.png"))
    
    # Histogram of Residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(model.resid, kde=True, bins=30)
    plt.xlabel("Residuals")
    plt.title("Distribution of Residuals")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residuals_histogram.png"))
    
    # Q-Q Plot for Residuals
    plt.figure(figsize=(10, 6))
    sm.qqplot(model.resid, line='45', fit=True)
    plt.title("Q-Q Plot of Residuals")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "qq_plot.png"))
    
    # Correlation Matrix
    plt.figure(figsize=(10, 8))
    corr_matrix = df_clean[predictors + ['Final_Score']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    
    print(f"\nAnalysis complete. Results saved to {output_dir} directory.")
    
    # Return the model for further analysis if needed
    return model, df_clean