"""
Statistical Analysis for Employee Dataset
Comprehensive descriptive, inferential, and exploratory statistical analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency, f_oneway, mannwhitneyu, shapiro, normaltest
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_root)

# Create results directories
os.makedirs('results/plots', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

print("="*80)
print("STATISTICAL ANALYSIS - EMPLOYEE DATASET")
print("="*80)

# Load cleaned dataset
df = pd.read_csv('data/processed/employees_cleaned.csv')

# =============================================================================
# 1. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n1. DESCRIPTIVE STATISTICS")
print("="*80)

numerical_cols = ['Salary', 'Bonus_pct', 'Years_of_Service']

# Basic descriptive statistics
print("\nBasic Descriptive Statistics:")
desc_stats = df[numerical_cols].describe()
print(desc_stats)
desc_stats.to_csv('results/tables/descriptive_statistics.csv')

# Additional descriptive statistics
print("\nAdditional Descriptive Statistics:")
additional_stats = pd.DataFrame({
    'Variable': numerical_cols,
    'Skewness': [stats.skew(df[col].dropna()) for col in numerical_cols],
    'Kurtosis': [stats.kurtosis(df[col].dropna()) for col in numerical_cols],
    'Variance': [df[col].var() for col in numerical_cols],
    'Coefficient of Variation': [df[col].std() / df[col].mean() * 100 for col in numerical_cols]
})
print(additional_stats)
additional_stats.to_csv('results/tables/additional_descriptive_statistics.csv')

# =============================================================================
# 2. NORMALITY TESTS
# =============================================================================
print("\n2. NORMALITY TESTS")
print("="*80)

normality_results = []

for col in numerical_cols:
    data = df[col].dropna()
    
    # Shapiro-Wilk test (for smaller samples)
    if len(data) <= 5000:
        stat_sw, p_value_sw = shapiro(data)
        normality_results.append({
            'Variable': col,
            'Test': 'Shapiro-Wilk',
            'Statistic': stat_sw,
            'P-value': p_value_sw,
            'Normal': 'Yes' if p_value_sw > 0.05 else 'No'
        })
    
    # D'Agostino's normality test
    stat_da, p_value_da = normaltest(data)
    normality_results.append({
        'Variable': col,
        'Test': "D'Agostino",
        'Statistic': stat_da,
        'P-value': p_value_da,
        'Normal': 'Yes' if p_value_da > 0.05 else 'No'
    })

normality_df = pd.DataFrame(normality_results)
print("\nNormality Test Results:")
print(normality_df)
normality_df.to_csv('results/tables/normality_tests.csv', index=False)

# Q-Q plots for normality
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(numerical_cols):
    stats.probplot(df[col].dropna(), dist="norm", plot=axes[i])
    axes[i].set_title(f'Q-Q Plot: {col}')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/qq_plots_normality.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 3. INFERENTIAL STATISTICS - T-TESTS
# =============================================================================
print("\n3. INFERENTIAL STATISTICS - T-TESTS")
print("="*80)

# T-test: Salary by Gender
print("\n3.1. T-test: Salary by Gender")
male_salary = df[df['Gender'] == 'Male']['Salary'].dropna()
female_salary = df[df['Gender'] == 'Female']['Salary'].dropna()

t_stat_gender, p_value_gender = ttest_ind(male_salary, female_salary)
print(f"T-statistic: {t_stat_gender:.4f}")
print(f"P-value: {p_value_gender:.4f}")
print(f"Mean Male Salary: ${male_salary.mean():,.2f}")
print(f"Mean Female Salary: ${female_salary.mean():,.2f}")
print(f"Significant difference: {'Yes' if p_value_gender < 0.05 else 'No'}")

# T-test: Salary by Senior Management
print("\n3.2. T-test: Salary by Senior Management")
sm_salary = df[df['Senior_Management'] == True]['Salary'].dropna()
non_sm_salary = df[df['Senior_Management'] == False]['Salary'].dropna()

t_stat_sm, p_value_sm = ttest_ind(sm_salary, non_sm_salary)
print(f"T-statistic: {t_stat_sm:.4f}")
print(f"P-value: {p_value_sm:.4f}")
print(f"Mean Senior Management Salary: ${sm_salary.mean():,.2f}")
print(f"Mean Non-Senior Management Salary: ${non_sm_salary.mean():,.2f}")
print(f"Significant difference: {'Yes' if p_value_sm < 0.05 else 'No'}")

# Save t-test results
ttest_results = pd.DataFrame([
    {'Test': 'Salary by Gender', 'T-statistic': t_stat_gender, 'P-value': p_value_gender, 'Significant': 'Yes' if p_value_gender < 0.05 else 'No'},
    {'Test': 'Salary by Senior Management', 'T-statistic': t_stat_sm, 'P-value': p_value_sm, 'Significant': 'Yes' if p_value_sm < 0.05 else 'No'}
])
ttest_results.to_csv('results/tables/ttest_results.csv', index=False)

# =============================================================================
# 4. NON-PARAMETRIC TESTS - MANN-WHITNEY U TEST
# =============================================================================
print("\n4. NON-PARAMETRIC TESTS - MANN-WHITNEY U TEST")
print("="*80)

# Mann-Whitney U test: Salary by Gender
print("\n4.1. Mann-Whitney U Test: Salary by Gender")
u_stat_gender, u_p_value_gender = mannwhitneyu(male_salary, female_salary, alternative='two-sided')
print(f"U-statistic: {u_stat_gender:.4f}")
print(f"P-value: {u_p_value_gender:.4f}")
print(f"Significant difference: {'Yes' if u_p_value_gender < 0.05 else 'No'}")

# Mann-Whitney U test: Salary by Senior Management
print("\n4.2. Mann-Whitney U Test: Salary by Senior Management")
u_stat_sm, u_p_value_sm = mannwhitneyu(sm_salary, non_sm_salary, alternative='two-sided')
print(f"U-statistic: {u_stat_sm:.4f}")
print(f"P-value: {u_p_value_sm:.4f}")
print(f"Significant difference: {'Yes' if u_p_value_sm < 0.05 else 'No'}")

# =============================================================================
# 5. CHI-SQUARE TESTS
# =============================================================================
print("\n5. CHI-SQUARE TESTS")
print("="*80)

# Chi-square test: Gender and Senior Management
print("\n5.1. Chi-square Test: Gender and Senior Management")
contingency_table = pd.crosstab(df['Gender'], df['Senior_Management'])
print("\nContingency Table:")
print(contingency_table)

chi2_stat, chi2_p_value, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-square statistic: {chi2_stat:.4f}")
print(f"P-value: {chi2_p_value:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"Significant association: {'Yes' if chi2_p_value < 0.05 else 'No'}")

# Chi-square test: Gender and Team (top teams)
print("\n5.2. Chi-square Test: Gender and Team (Top 5 Teams)")
top_teams = df['Team'].value_counts().head(5).index
df_top_teams = df[df['Team'].isin(top_teams)]
contingency_table_team = pd.crosstab(df_top_teams['Gender'], df_top_teams['Team'])
print("\nContingency Table:")
print(contingency_table_team)

chi2_stat_team, chi2_p_value_team, dof_team, expected_team = chi2_contingency(contingency_table_team)
print(f"\nChi-square statistic: {chi2_stat_team:.4f}")
print(f"P-value: {chi2_p_value_team:.4f}")
print(f"Degrees of freedom: {dof_team}")
print(f"Significant association: {'Yes' if chi2_p_value_team < 0.05 else 'No'}")

# Save chi-square results
chisquare_results = pd.DataFrame([
    {'Test': 'Gender and Senior Management', 'Chi-square': chi2_stat, 'P-value': chi2_p_value, 'Significant': 'Yes' if chi2_p_value < 0.05 else 'No'},
    {'Test': 'Gender and Team', 'Chi-square': chi2_stat_team, 'P-value': chi2_p_value_team, 'Significant': 'Yes' if chi2_p_value_team < 0.05 else 'No'}
])
chisquare_results.to_csv('results/tables/chisquare_results.csv', index=False)

# =============================================================================
# 6. ANOVA - ANALYSIS OF VARIANCE
# =============================================================================
print("\n6. ANOVA - ANALYSIS OF VARIANCE")
print("="*80)

# ANOVA: Salary across Teams (top 10 teams)
print("\n6.1. ANOVA: Salary across Teams (Top 10 Teams)")
top_teams_anova = df['Team'].value_counts().head(10).index
df_anova = df[df['Team'].isin(top_teams_anova)]

team_groups = [df_anova[df_anova['Team'] == team]['Salary'].dropna() for team in top_teams_anova]
f_stat, p_value_anova = f_oneway(*team_groups)

print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value_anova:.4f}")
print(f"Significant difference: {'Yes' if p_value_anova < 0.05 else 'No'}")

# Post-hoc analysis: Pairwise comparisons
print("\n6.2. Post-hoc Analysis: Pairwise T-tests (Bonferroni correction)")
from scipy.stats import ttest_ind
from itertools import combinations
import numpy as np

pairwise_results = []
alpha = 0.05
n_comparisons = len(list(combinations(top_teams_anova, 2)))
bonferroni_alpha = alpha / n_comparisons

for team1, team2 in combinations(top_teams_anova, 2):
    group1 = df_anova[df_anova['Team'] == team1]['Salary'].dropna()
    group2 = df_anova[df_anova['Team'] == team2]['Salary'].dropna()
    
    if len(group1) > 0 and len(group2) > 0:
        t_stat, p_val = ttest_ind(group1, group2)
        pairwise_results.append({
            'Team1': team1,
            'Team2': team2,
            'T-statistic': t_stat,
            'P-value': p_val,
            'Significant (Bonferroni)': 'Yes' if p_val < bonferroni_alpha else 'No'
        })

pairwise_df = pd.DataFrame(pairwise_results)
print(f"\nNumber of comparisons: {n_comparisons}")
print(f"Bonferroni corrected alpha: {bonferroni_alpha:.6f}")
print("\nSignificant pairwise differences:")
print(pairwise_df[pairwise_df['Significant (Bonferroni)'] == 'Yes'])
pairwise_df.to_csv('results/tables/anova_pairwise_results.csv', index=False)

# =============================================================================
# 7. CORRELATION ANALYSIS WITH SIGNIFICANCE TESTING
# =============================================================================
print("\n7. CORRELATION ANALYSIS WITH SIGNIFICANCE TESTING")
print("="*80)

# Pearson correlation with significance
correlation_results = []
for i, col1 in enumerate(numerical_cols):
    for col2 in numerical_cols[i+1:]:
        data1 = df[col1].dropna()
        data2 = df[col2].dropna()
        
        # Align data
        common_idx = data1.index.intersection(data2.index)
        data1_aligned = data1[common_idx]
        data2_aligned = data2[common_idx]
        
        if len(data1_aligned) > 2:
            corr_coef, p_value = stats.pearsonr(data1_aligned, data2_aligned)
            correlation_results.append({
                'Variable1': col1,
                'Variable2': col2,
                'Correlation': corr_coef,
                'P-value': p_value,
                'Significant': 'Yes' if p_value < 0.05 else 'No'
            })

corr_df = pd.DataFrame(correlation_results)
print("\nCorrelation Analysis Results:")
print(corr_df)
corr_df.to_csv('results/tables/correlation_analysis.csv', index=False)

# Visualize correlations
plt.figure(figsize=(10, 8))
corr_matrix = df[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.3f')
plt.title('Correlation Matrix with Significance Testing')
plt.tight_layout()
plt.savefig('results/plots/correlation_matrix_statistical.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 8. CONFIDENCE INTERVALS
# =============================================================================
print("\n8. CONFIDENCE INTERVALS")
print("="*80)

confidence_intervals = []
confidence_level = 0.95
alpha = 1 - confidence_level

for col in numerical_cols:
    data = df[col].dropna()
    n = len(data)
    mean = data.mean()
    std_err = stats.sem(data)
    margin_error = stats.t.interval(confidence_level, n-1, loc=mean, scale=std_err)
    
    confidence_intervals.append({
        'Variable': col,
        'Mean': mean,
        'Std Error': std_err,
        'Lower CI (95%)': margin_error[0],
        'Upper CI (95%)': margin_error[1],
        'Margin of Error': (margin_error[1] - margin_error[0]) / 2
    })

ci_df = pd.DataFrame(confidence_intervals)
print("\n95% Confidence Intervals:")
print(ci_df)
ci_df.to_csv('results/tables/confidence_intervals.csv', index=False)

# Visualize confidence intervals
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(numerical_cols):
    data = df[col].dropna()
    mean = data.mean()
    ci_lower = ci_df[ci_df['Variable'] == col]['Lower CI (95%)'].values[0]
    ci_upper = ci_df[ci_df['Variable'] == col]['Upper CI (95%)'].values[0]
    
    axes[i].barh([0], [mean], xerr=[[mean - ci_lower], [ci_upper - mean]], 
                 capsize=10, color='steelblue', alpha=0.7)
    axes[i].axvline(mean, color='red', linestyle='--', linewidth=2, label='Mean')
    axes[i].set_title(f'95% CI for {col}')
    axes[i].set_xlabel(col)
    axes[i].set_yticks([])
    axes[i].legend()
    axes[i].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('results/plots/confidence_intervals.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 9. SUMMARY OF STATISTICAL TESTS
# =============================================================================
print("\n9. SUMMARY OF STATISTICAL TESTS")
print("="*80)

summary = pd.DataFrame([
    {'Test Type': 'T-test', 'Hypothesis': 'Salary differs by Gender', 'Result': 'Significant' if p_value_gender < 0.05 else 'Not Significant'},
    {'Test Type': 'T-test', 'Hypothesis': 'Salary differs by Senior Management', 'Result': 'Significant' if p_value_sm < 0.05 else 'Not Significant'},
    {'Test Type': 'Mann-Whitney U', 'Hypothesis': 'Salary differs by Gender (non-parametric)', 'Result': 'Significant' if u_p_value_gender < 0.05 else 'Not Significant'},
    {'Test Type': 'Chi-square', 'Hypothesis': 'Association between Gender and Senior Management', 'Result': 'Significant' if chi2_p_value < 0.05 else 'Not Significant'},
    {'Test Type': 'ANOVA', 'Hypothesis': 'Salary differs across Teams', 'Result': 'Significant' if p_value_anova < 0.05 else 'Not Significant'},
])

print("\nStatistical Test Summary:")
print(summary)
summary.to_csv('results/tables/statistical_tests_summary.csv', index=False)

print("\n" + "="*80)
print("STATISTICAL ANALYSIS COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\nResults saved in:")
print("- Tables: results/tables/")
print("- Plots: results/plots/")
