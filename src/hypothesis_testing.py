import pandas as pd
import scipy.stats as stats

def perform_hypothesis_testing(df):
    print("\nStarting Hypothesis Testing...\n")
    
    # Hypothesis 1: Do male and female students have significantly different final scores?
    males = df[df['Gender'] == 'Male']['Final_Score'].dropna()
    females = df[df['Gender'] == 'Female']['Final_Score'].dropna()
    
    t_stat, p_value = stats.ttest_ind(males, females, equal_var=False)  # Welch’s t-test
    print("T-test for difference in Final Scores between Genders:")
    print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Conclusion: Reject the null hypothesis. There is a significant difference in Final Scores between male and female students.")
    else:
        print("Conclusion: Fail to reject the null hypothesis. No significant difference in Final Scores between male and female students.")
    
    # Hypothesis 2: Is there a relationship between Internet Access at Home and Grade?
    contingency_table = pd.crosstab(df['Internet_Access_at_Home'], df['Grade'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    
    print("\nChi-Square Test for Association between Internet Access at Home and Grade:")
    print(f"Chi-Square Statistic: {chi2:.4f}, p-value: {p:.4f}")
    if p < 0.05:
        print("Conclusion: Reject the null hypothesis. There is a significant association between Internet Access at Home and Grade.")
    else:
        print("Conclusion: Fail to reject the null hypothesis. No significant association between Internet Access at Home and Grade.")
    
    print("\nHypothesis Testing Completed.\n")
    
    return {
        "t-test": {"t_statistic": t_stat, "p_value": p_value},
        "chi-square": {"chi2_statistic": chi2, "p_value": p}
    }

if __name__ == "__main__":
    df = pd.read_csv("data/Students_Grading_Dataset.csv")
    perform_hypothesis_testing(df)
