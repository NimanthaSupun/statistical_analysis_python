import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
import seaborn as sns

def perform_hypothesis_testing(df, output_dir=None):
    print("\n" + "="*50)
    print("HYPOTHESIS TESTING ANALYSIS")
    print("="*50)
    
    # Dataset description
    print("\nDataset Description:")
    print(f"Source: Student academic performance dataset")
    print(f"Size: {df.shape[0]} students, {df.shape[1]} variables")
    print(f"Variables include: Demographics, academic scores, study habits, and socioeconomic factors")
    print(f"Date analyzed: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    
    results = {}
    
    # Create figures directory if needed
    if output_dir and not os.path.exists(os.path.join(output_dir, "figures")):
        os.makedirs(os.path.join(output_dir, "figures"))
    
    # Hypothesis 1: Gender differences in Final Scores (t-test)
    print("\n" + "-"*50)
    print("Hypothesis 1: Gender differences in Final Scores")
    print("-"*50)
    
    print("Null Hypothesis (H0): There is no significant difference in Final Scores between male and female students.")
    print("Alternative Hypothesis (H1): There is a significant difference in Final Scores between male and female students.")
    print("\nMethodology: Independent samples t-test (Welch's t-test for unequal variances)")
    
    males = df[df['Gender'] == 'Male']['Final_Score'].dropna()
    females = df[df['Gender'] == 'Female']['Final_Score'].dropna()
    
    # Check normality assumptions
    _, p_norm_male = stats.shapiro(males) if len(males) < 5000 else (0, 0.05)
    _, p_norm_female = stats.shapiro(females) if len(females) < 5000 else (0, 0.05)
    normality_met = p_norm_male > 0.05 and p_norm_female > 0.05
    
    print(f"\nChecking assumptions:")
    print(f"- Normality test (Shapiro-Wilk) for males: p-value = {p_norm_male:.4f} ({'Satisfied' if p_norm_male > 0.05 else 'Not satisfied'})")
    print(f"- Normality test (Shapiro-Wilk) for females: p-value = {p_norm_female:.4f} ({'Satisfied' if p_norm_female > 0.05 else 'Not satisfied'})")
    
    # Create visualization
    if output_dir:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Gender', y='Final_Score', data=df)
        plt.title('Final Scores by Gender')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figures", "gender_final_scores.png"))
        plt.close()
    
    # Conduct appropriate test based on normality
    if normality_met:
        # If normality assumptions are met, use t-test
        t_stat, p_value = stats.ttest_ind(males, females, equal_var=False)  # Welch's t-test
        mean_diff = males.mean() - females.mean()
        
        # Calculate confidence interval manually
        n1, n2 = len(males), len(females)
        std1, std2 = males.std(), females.std()
        se = np.sqrt((std1**2/n1) + (std2**2/n2))
        df_welch = ((std1**2/n1 + std2**2/n2)**2) / ((std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1))
        t_crit = stats.t.ppf(0.975, df_welch)
        ci_lower = mean_diff - t_crit * se
        ci_upper = mean_diff + t_crit * se
        
        test_name = "Welch's t-test"
        test_stat_name = "t-statistic"
        
    else:
        # If normality assumptions are violated, use Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(males, females, alternative='two-sided')
        mean_diff = males.median() - females.median()  # Using median for non-parametric
        ci_lower, ci_upper = "N/A", "N/A"  # Mann-Whitney doesn't provide CI directly
        
        test_name = "Mann-Whitney U test"
        test_stat_name = "U-statistic"
        t_stat = u_stat  # Store U stat in t_stat for consistency in results dict
    
    print(f"\nTest performed: {test_name}")
    print(f"{test_stat_name}: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    if test_name == "Welch's t-test":
        print(f"Mean difference (Male - Female): {mean_diff:.4f}")
        print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    else:
        print(f"Median difference (Male - Female): {mean_diff:.4f}")
    
    print("\nResults interpretation:")
    if p_value < 0.05:
        conclusion = "Reject the null hypothesis. There is a significant difference in Final Scores between male and female students."
    else:
        conclusion = "Fail to reject the null hypothesis. No significant difference in Final Scores between male and female students."
    
    print(conclusion)
    
    # Store results
    results["gender_final_scores"] = {
        "test_type": test_name,
        "test_statistic": t_stat,
        "p_value": p_value,
        "mean_difference": mean_diff,
        "ci_lower": ci_lower if test_name == "Welch's t-test" else None,
        "ci_upper": ci_upper if test_name == "Welch's t-test" else None,
        "conclusion": conclusion
    }
    
    # Hypothesis 2: Association between Internet Access and Grades (Chi-Square)
    print("\n" + "-"*50)
    print("Hypothesis 2: Association between Internet Access and Grades")
    print("-"*50)
    
    print("Null Hypothesis (H0): There is no association between Internet Access at Home and Grade.")
    print("Alternative Hypothesis (H1): There is an association between Internet Access at Home and Grade.")
    print("\nMethodology: Chi-Square Test of Independence")
    
    # Create a contingency table
    contingency_table = pd.crosstab(df['Internet_Access_at_Home'], df['Grade'])
    print("\nContingency Table:")
    print(contingency_table)
    
    # Check assumptions
    expected = stats.contingency.expected_freq(contingency_table)
    assumptions_met = np.all(expected >= 5)
    
    print(f"\nChecking assumptions:")
    print(f"- All expected frequencies ≥ 5: {'Satisfied' if assumptions_met else 'Not satisfied'}")
    
    if not assumptions_met:
        print("Warning: Some expected frequencies are less than 5, chi-square results may not be reliable.")
    
    # Run chi-square test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Create visualization
    if output_dir:
        plt.figure(figsize=(10, 6))
        pd.crosstab(df['Internet_Access_at_Home'], df['Grade'], normalize='index').plot(kind='bar', stacked=True)
        plt.title('Grade Distribution by Internet Access at Home')
        plt.xlabel('Internet Access at Home')
        plt.ylabel('Proportion')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figures", "internet_access_grades.png"))
        plt.close()
    
    print(f"\nTest performed: Chi-Square Test of Independence")
    print(f"Chi-Square statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"p-value: {p:.4f}")
    
    print("\nResults interpretation:")
    if p < 0.05:
        conclusion = "Reject the null hypothesis. There is a significant association between Internet Access at Home and Grade."
    else:
        conclusion = "Fail to reject the null hypothesis. No significant association between Internet Access at Home and Grade."
    
    print(conclusion)
    
    # Store results
    results["internet_access_grades"] = {
        "test_type": "Chi-Square Test of Independence",
        "chi2_statistic": chi2,
        "degrees_of_freedom": dof,
        "p_value": p,
        "conclusion": conclusion
    }
    
    # Hypothesis 3: Correlation between Study Hours and Total Score (Pearson/Spearman)
    print("\n" + "-"*50)
    print("Hypothesis 3: Correlation between Study Hours and Total Score")
    print("-"*50)
    
    print("Null Hypothesis (H0): There is no correlation between Study Hours per Week and Total Score.")
    print("Alternative Hypothesis (H1): There is a correlation between Study Hours per Week and Total Score.")
    print("\nMethodology: Correlation analysis (Pearson/Spearman)")
    
    # Check normality for both variables
    _, p_norm_hours = stats.shapiro(df['Study_Hours_per_Week'].dropna()) if len(df['Study_Hours_per_Week'].dropna()) < 5000 else (0, 0.01)
    _, p_norm_score = stats.shapiro(df['Total_Score'].dropna()) if len(df['Total_Score'].dropna()) < 5000 else (0, 0.01)
    normality_met = p_norm_hours > 0.05 and p_norm_score > 0.05
    
    print(f"\nChecking assumptions:")
    print(f"- Normality test for Study Hours: p-value = {p_norm_hours:.4f} ({'Satisfied' if p_norm_hours > 0.05 else 'Not satisfied'})")
    print(f"- Normality test for Total Score: p-value = {p_norm_score:.4f} ({'Satisfied' if p_norm_score > 0.05 else 'Not satisfied'})")
    
    # Create visualization
    if output_dir:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Study_Hours_per_Week', y='Total_Score', data=df)
        plt.title('Relationship between Study Hours and Total Score')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figures", "study_hours_total_score.png"))
        plt.close()
    
    # Conduct appropriate correlation test
    if normality_met:
        # Pearson correlation for normally distributed data
        r, p_value = stats.pearsonr(df['Study_Hours_per_Week'].dropna(), df['Total_Score'].dropna())
        test_name = "Pearson correlation"
    else:
        # Spearman correlation for non-normally distributed data
        r, p_value = stats.spearmanr(df['Study_Hours_per_Week'].dropna(), df['Total_Score'].dropna())
        test_name = "Spearman rank correlation"
    
    print(f"\nTest performed: {test_name}")
    print(f"Correlation coefficient: {r:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    print("\nResults interpretation:")
    if p_value < 0.05:
        if r > 0:
            strength = "positive"
        else:
            strength = "negative"
            
        if abs(r) < 0.3:
            magnitude = "weak"
        elif abs(r) < 0.7:
            magnitude = "moderate"
        else:
            magnitude = "strong"
            
        conclusion = f"Reject the null hypothesis. There is a statistically significant {magnitude} {strength} correlation between Study Hours per Week and Total Score (r = {r:.4f})."
    else:
        conclusion = "Fail to reject the null hypothesis. No significant correlation between Study Hours per Week and Total Score."
    
    print(conclusion)
    
    # Store results
    results["study_hours_total_score"] = {
        "test_type": test_name,
        "correlation_coefficient": r,
        "p_value": p_value,
        "conclusion": conclusion
    }
    
    # Hypothesis 4: ANOVA - Department differences in Midterm Scores
    print("\n" + "-"*50)
    print("Hypothesis 4: Department differences in Midterm Scores")
    print("-"*50)
    
    print("Null Hypothesis (H0): There is no difference in Midterm Scores across departments.")
    print("Alternative Hypothesis (H1): At least one department has different Midterm Scores.")
    print("\nMethodology: One-way ANOVA (or Kruskal-Wallis H-test)")
    
    # Prepare data
    dept_groups = []
    dept_names = []
    
    for dept in df['Department'].unique():
        scores = df[df['Department'] == dept]['Midterm_Score'].dropna()
        if len(scores) > 0:  # Only include if there are values
            dept_groups.append(scores)
            dept_names.append(dept)
    
    # Check normality and equal variances
    normality_results = []
    for i, scores in enumerate(dept_groups):
        if len(scores) < 5000:  # Shapiro-Wilk works for smaller samples
            _, p_norm = stats.shapiro(scores)
            normality_results.append(p_norm > 0.05)
        else:
            normality_results.append(False)  # For large samples, assume non-normality
    
    equal_var = stats.levene(*dept_groups)[1] > 0.05
    
    print(f"\nChecking assumptions:")
    print(f"- Normality across departments: {'All satisfied' if all(normality_results) else 'Not all satisfied'}")
    print(f"- Equal variances (Levene's test): p-value = {stats.levene(*dept_groups)[1]:.4f} ({'Satisfied' if equal_var else 'Not satisfied'})")
    
    # Create visualization
    if output_dir:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Department', y='Midterm_Score', data=df)
        plt.title('Midterm Scores by Department')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figures", "department_midterm_scores.png"))
        plt.close()
    
    # Conduct appropriate test
    if all(normality_results) and equal_var:
        # One-way ANOVA for normal data with equal variances
        f_stat, p_value = stats.f_oneway(*dept_groups)
        test_name = "One-way ANOVA"
        test_stat_name = "F-statistic"
        stat = f_stat
    else:
        # Kruskal-Wallis H-test for non-normal data or unequal variances
        h_stat, p_value = stats.kruskal(*dept_groups)
        test_name = "Kruskal-Wallis H-test"
        test_stat_name = "H-statistic"
        stat = h_stat
    
    print(f"\nTest performed: {test_name}")
    print(f"{test_stat_name}: {stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    # Post-hoc tests if significant
    posthoc_results = None
    if p_value < 0.05:
        if test_name == "One-way ANOVA":
            print("\nPost-hoc analysis: Tukey's HSD")
            # Prepare data for Tukey's HSD
            data_for_posthoc = df[['Department', 'Midterm_Score']].dropna()
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            posthoc = pairwise_tukeyhsd(data_for_posthoc['Midterm_Score'], data_for_posthoc['Department'], alpha=0.05)
            posthoc_results = posthoc
            print(posthoc)
        else:
            print("\nPost-hoc analysis: Dunn's test with Bonferroni correction")
            from scikit_posthocs import posthoc_dunn
            posthoc_matrix = posthoc_dunn(dept_groups, p_adjust='bonferroni')
            posthoc_results = posthoc_matrix
            print("Dunn's test p-values (adjusted with Bonferroni method):")
            print(pd.DataFrame(posthoc_matrix, index=dept_names, columns=dept_names))
    
    print("\nResults interpretation:")
    if p_value < 0.05:
        conclusion = f"Reject the null hypothesis. There are significant differences in Midterm Scores between departments."
        if posthoc_results is not None:
            conclusion += " See post-hoc analysis for specific group differences."
    else:
        conclusion = "Fail to reject the null hypothesis. No significant differences in Midterm Scores between departments."
    
    print(conclusion)
    
    # Store results
    results["department_midterm_scores"] = {
        "test_type": test_name,
        "test_statistic": stat,
        "p_value": p_value,
        "conclusion": conclusion,
        "posthoc_results": posthoc_results
    }
    
    # Save results if output directory is provided
    if output_dir:
        save_results_to_file(results, output_dir)
    
    print("\nHypothesis Testing Completed.\n")
    
    return results

def save_results_to_file(results, output_dir):
    """
    Save hypothesis testing results to a text file
    """
    with open(os.path.join(output_dir, "hypothesis_results.txt"), "w") as f:
        f.write("===================================\n")
        f.write("HYPOTHESIS TESTING RESULTS SUMMARY\n")
        f.write("===================================\n\n")
        
        # Hypothesis 1: Gender differences in Final Scores
        f.write("HYPOTHESIS 1: Gender differences in Final Scores\n")
        f.write("-" * 50 + "\n")
        f.write("Null Hypothesis (H0): There is no significant difference in Final Scores between male and female students.\n")
        f.write("Alternative Hypothesis (H1): There is a significant difference in Final Scores between male and female students.\n\n")
        f.write(f"Test: {results['gender_final_scores']['test_type']}\n")
        f.write(f"Test statistic: {results['gender_final_scores']['test_statistic']:.4f}\n")
        f.write(f"p-value: {results['gender_final_scores']['p_value']:.4f}\n")
        
        if 'ci_lower' in results['gender_final_scores'] and results['gender_final_scores']['ci_lower'] is not None:
            f.write(f"Mean difference: {results['gender_final_scores']['mean_difference']:.4f}\n")
            f.write(f"95% Confidence Interval: [{results['gender_final_scores']['ci_lower']:.4f}, {results['gender_final_scores']['ci_upper']:.4f}]\n")
        else:
            f.write(f"Median difference: {results['gender_final_scores']['mean_difference']:.4f}\n")
        
        f.write(f"Conclusion: {results['gender_final_scores']['conclusion']}\n\n")
        
        # Hypothesis 2: Internet Access and Grades
        f.write("HYPOTHESIS 2: Association between Internet Access and Grades\n")
        f.write("-" * 50 + "\n")
        f.write("Null Hypothesis (H0): There is no association between Internet Access at Home and Grade.\n")
        f.write("Alternative Hypothesis (H1): There is an association between Internet Access at Home and Grade.\n\n")
        f.write(f"Test: {results['internet_access_grades']['test_type']}\n")
        f.write(f"Chi-Square statistic: {results['internet_access_grades']['chi2_statistic']:.4f}\n")
        f.write(f"Degrees of freedom: {results['internet_access_grades']['degrees_of_freedom']}\n")
        f.write(f"p-value: {results['internet_access_grades']['p_value']:.4f}\n")
        f.write(f"Conclusion: {results['internet_access_grades']['conclusion']}\n\n")
        
        # Hypothesis 3: Study Hours and Total Score
        f.write("HYPOTHESIS 3: Correlation between Study Hours and Total Score\n")
        f.write("-" * 50 + "\n")
        f.write("Null Hypothesis (H0): There is no correlation between Study Hours per Week and Total Score.\n")
        f.write("Alternative Hypothesis (H1): There is a correlation between Study Hours per Week and Total Score.\n\n")
        f.write(f"Test: {results['study_hours_total_score']['test_type']}\n")
        f.write(f"Correlation coefficient: {results['study_hours_total_score']['correlation_coefficient']:.4f}\n")
        f.write(f"p-value: {results['study_hours_total_score']['p_value']:.4f}\n")
        f.write(f"Conclusion: {results['study_hours_total_score']['conclusion']}\n\n")
        
        # Hypothesis 4: Department differences in Midterm Scores
        f.write("HYPOTHESIS 4: Department differences in Midterm Scores\n")
        f.write("-" * 50 + "\n")
        f.write("Null Hypothesis (H0): There is no difference in Midterm Scores across departments.\n")
        f.write("Alternative Hypothesis (H1): At least one department has different Midterm Scores.\n\n")
        f.write(f"Test: {results['department_midterm_scores']['test_type']}\n")
        f.write(f"Test statistic: {results['department_midterm_scores']['test_statistic']:.4f}\n")
        f.write(f"p-value: {results['department_midterm_scores']['p_value']:.4f}\n")
        f.write(f"Conclusion: {results['department_midterm_scores']['conclusion']}\n")
        
        # Final note
        f.write("\n\nNote: This analysis was performed on student academic performance data. The results should be interpreted in the context of educational research and might be influenced by sample characteristics, data quality, and other confounding factors.\n")

if __name__ == "__main__":
    # For testing the module directly
    df = pd.read_csv("data/Students_Grading_Dataset.csv")
    perform_hypothesis_testing(df, "output")