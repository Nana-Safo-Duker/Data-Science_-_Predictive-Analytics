"""
Univariate, Bivariate, and Multivariate Analysis for Employee Dataset
Comprehensive analysis of individual variables, pairs of variables, and multiple variables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
print("UNIVARIATE, BIVARIATE, AND MULTIVARIATE ANALYSIS - EMPLOYEE DATASET")
print("="*80)

# Load cleaned dataset
df = pd.read_csv('data/processed/employees_cleaned.csv')

numerical_cols = ['Salary', 'Bonus_pct', 'Years_of_Service']
categorical_cols = ['Gender', 'Senior_Management', 'Team']

# =============================================================================
# 1. UNIVARIATE ANALYSIS
# =============================================================================
print("\n1. UNIVARIATE ANALYSIS")
print("="*80)

# 1.1 Numerical Variables
print("\n1.1. Numerical Variables Univariate Analysis")

for col in numerical_cols:
    print(f"\n--- {col} ---")
    data = df[col].dropna()
    print(f"Mean: {data.mean():.2f}")
    print(f"Median: {data.median():.2f}")
    print(f"Mode: {data.mode().values[0] if len(data.mode()) > 0 else 'N/A'}")
    print(f"Standard Deviation: {data.std():.2f}")
    print(f"Variance: {data.var():.2f}")
    print(f"Skewness: {stats.skew(data):.2f}")
    print(f"Kurtosis: {stats.kurtosis(data):.2f}")
    print(f"Min: {data.min():.2f}")
    print(f"Max: {data.max():.2f}")
    print(f"Range: {data.max() - data.min():.2f}")
    print(f"Q1: {data.quantile(0.25):.2f}")
    print(f"Q3: {data.quantile(0.75):.2f}")
    print(f"IQR: {data.quantile(0.75) - data.quantile(0.25):.2f}")

# Visualizations for numerical variables
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

for i, col in enumerate(numerical_cols):
    data = df[col].dropna()
    
    # Histogram
    axes[i, 0].hist(data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[i, 0].set_title(f'Histogram: {col}')
    axes[i, 0].set_xlabel(col)
    axes[i, 0].set_ylabel('Frequency')
    axes[i, 0].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
    axes[i, 0].axvline(data.median(), color='green', linestyle='--', label=f'Median: {data.median():.2f}')
    axes[i, 0].legend()
    axes[i, 0].grid(True, alpha=0.3)
    
    # Box plot
    axes[i, 1].boxplot(data, vert=True)
    axes[i, 1].set_title(f'Box Plot: {col}')
    axes[i, 1].set_ylabel(col)
    axes[i, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(data, dist="norm", plot=axes[i, 2])
    axes[i, 2].set_title(f'Q-Q Plot: {col}')
    axes[i, 2].grid(True, alpha=0.3)
    
    # Violin plot
    axes[i, 3].violinplot([data], positions=[0], showmeans=True, showmedians=True)
    axes[i, 3].set_title(f'Violin Plot: {col}')
    axes[i, 3].set_ylabel(col)
    axes[i, 3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/univariate_numerical.png', dpi=300, bbox_inches='tight')
plt.close()

# 1.2 Categorical Variables
print("\n1.2. Categorical Variables Univariate Analysis")

for col in categorical_cols:
    print(f"\n--- {col} ---")
    counts = df[col].value_counts()
    percentages = df[col].value_counts(normalize=True) * 100
    print("Counts:")
    print(counts)
    print("\nPercentages:")
    print(percentages)
    print(f"Mode: {df[col].mode().values[0] if len(df[col].mode()) > 0 else 'N/A'}")
    print(f"Number of unique values: {df[col].nunique()}")

# Visualizations for categorical variables
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, col in enumerate(categorical_cols):
    if col == 'Team':
        # Show top 10 teams
        top_10 = df[col].value_counts().head(10)
        axes[i].barh(top_10.index, top_10.values, color='steelblue')
        axes[i].set_title(f'Top 10 {col} Distribution')
        axes[i].set_xlabel('Count')
    else:
        counts = df[col].value_counts()
        axes[i].bar(counts.index.astype(str), counts.values, color='steelblue', edgecolor='black')
        axes[i].set_title(f'{col} Distribution')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/plots/univariate_categorical.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 2. BIVARIATE ANALYSIS
# =============================================================================
print("\n2. BIVARIATE ANALYSIS")
print("="*80)

# 2.1 Numerical vs Numerical
print("\n2.1. Numerical vs Numerical Analysis")

# Correlation analysis
correlation_matrix = df[numerical_cols].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)
correlation_matrix.to_csv('results/tables/bivariate_correlation_matrix.csv')

# Scatter plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Salary vs Bonus %
axes[0].scatter(df['Salary'], df['Bonus_pct'], alpha=0.5, s=50)
axes[0].set_xlabel('Salary')
axes[0].set_ylabel('Bonus %')
axes[0].set_title('Salary vs Bonus %')
corr_sb = df['Salary'].corr(df['Bonus_pct'])
axes[0].text(0.05, 0.95, f'Correlation: {corr_sb:.3f}', transform=axes[0].transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[0].grid(True, alpha=0.3)

# Salary vs Years of Service
axes[1].scatter(df['Years_of_Service'], df['Salary'], alpha=0.5, s=50)
axes[1].set_xlabel('Years of Service')
axes[1].set_ylabel('Salary')
axes[1].set_title('Salary vs Years of Service')
corr_ys = df['Salary'].corr(df['Years_of_Service'])
axes[1].text(0.05, 0.95, f'Correlation: {corr_ys:.3f}', transform=axes[1].transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1].grid(True, alpha=0.3)

# Bonus % vs Years of Service
axes[2].scatter(df['Years_of_Service'], df['Bonus_pct'], alpha=0.5, s=50)
axes[2].set_xlabel('Years of Service')
axes[2].set_ylabel('Bonus %')
axes[2].set_title('Bonus % vs Years of Service')
corr_by = df['Bonus_pct'].corr(df['Years_of_Service'])
axes[2].text(0.05, 0.95, f'Correlation: {corr_by:.3f}', transform=axes[2].transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/bivariate_numerical_numerical.png', dpi=300, bbox_inches='tight')
plt.close()

# 2.2 Numerical vs Categorical
print("\n2.2. Numerical vs Categorical Analysis")

# Salary by Gender
print("\nSalary by Gender:")
print(df.groupby('Gender')['Salary'].agg(['mean', 'median', 'std', 'count']))

# Salary by Senior Management
print("\nSalary by Senior Management:")
print(df.groupby('Senior_Management')['Salary'].agg(['mean', 'median', 'std', 'count']))

# Salary by Team (top 10)
print("\nSalary by Team (Top 10):")
top_teams = df['Team'].value_counts().head(10).index
print(df[df['Team'].isin(top_teams)].groupby('Team')['Salary'].agg(['mean', 'median', 'std', 'count']))

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Salary by Gender - Box plot
df.boxplot(column='Salary', by='Gender', ax=axes[0, 0])
axes[0, 0].set_title('Salary Distribution by Gender')
axes[0, 0].set_xlabel('Gender')
axes[0, 0].set_ylabel('Salary')
axes[0, 0].grid(True, alpha=0.3)

# Salary by Gender - Violin plot
sns.violinplot(x='Gender', y='Salary', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Salary Distribution by Gender (Violin Plot)')
axes[0, 1].grid(True, alpha=0.3)

# Salary by Senior Management - Box plot
df.boxplot(column='Salary', by='Senior_Management', ax=axes[0, 2])
axes[0, 2].set_title('Salary Distribution by Senior Management')
axes[0, 2].set_xlabel('Senior Management')
axes[0, 2].set_ylabel('Salary')
axes[0, 2].grid(True, alpha=0.3)

# Salary by Team (top 10) - Box plot
top_teams_data = df[df['Team'].isin(top_teams)]
top_teams_data.boxplot(column='Salary', by='Team', ax=axes[1, 0])
axes[1, 0].set_title('Salary Distribution by Team (Top 10)')
axes[1, 0].set_xlabel('Team')
axes[1, 0].set_ylabel('Salary')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3)

# Bonus % by Gender
df.boxplot(column='Bonus_pct', by='Gender', ax=axes[1, 1])
axes[1, 1].set_title('Bonus % Distribution by Gender')
axes[1, 1].set_xlabel('Gender')
axes[1, 1].set_ylabel('Bonus %')
axes[1, 1].grid(True, alpha=0.3)

# Years of Service by Senior Management
df.boxplot(column='Years_of_Service', by='Senior_Management', ax=axes[1, 2])
axes[1, 2].set_title('Years of Service by Senior Management')
axes[1, 2].set_xlabel('Senior Management')
axes[1, 2].set_ylabel('Years of Service')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/bivariate_numerical_categorical.png', dpi=300, bbox_inches='tight')
plt.close()

# 2.3 Categorical vs Categorical
print("\n2.3. Categorical vs Categorical Analysis")

# Gender vs Senior Management
print("\nGender vs Senior Management:")
contingency_gender_sm = pd.crosstab(df['Gender'], df['Senior_Management'], margins=True)
print(contingency_gender_sm)
contingency_gender_sm.to_csv('results/tables/contingency_gender_senior_management.csv')

# Gender vs Team (top 10)
print("\nGender vs Team (Top 10):")
contingency_gender_team = pd.crosstab(df[df['Team'].isin(top_teams)]['Gender'], 
                                       df[df['Team'].isin(top_teams)]['Team'], margins=True)
print(contingency_gender_team)
contingency_gender_team.to_csv('results/tables/contingency_gender_team.csv')

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gender vs Senior Management - Stacked bar chart
contingency_gender_sm_plot = pd.crosstab(df['Gender'], df['Senior_Management'])
contingency_gender_sm_plot.plot(kind='bar', stacked=True, ax=axes[0], color=['lightcoral', 'lightgreen'])
axes[0].set_title('Gender vs Senior Management')
axes[0].set_xlabel('Gender')
axes[0].set_ylabel('Count')
axes[0].legend(title='Senior Management')
axes[0].tick_params(axis='x', rotation=0)
axes[0].grid(True, alpha=0.3, axis='y')

# Gender vs Team (top 5) - Heatmap
contingency_gender_team_plot = pd.crosstab(df[df['Team'].isin(top_teams[:5])]['Gender'], 
                                            df[df['Team'].isin(top_teams[:5])]['Team'])
sns.heatmap(contingency_gender_team_plot, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1])
axes[1].set_title('Gender vs Team (Top 5) - Heatmap')
axes[1].set_xlabel('Team')
axes[1].set_ylabel('Gender')

plt.tight_layout()
plt.savefig('results/plots/bivariate_categorical_categorical.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 3. MULTIVARIATE ANALYSIS
# =============================================================================
print("\n3. MULTIVARIATE ANALYSIS")
print("="*80)

# 3.1 Pairwise Relationships
print("\n3.1. Pairwise Relationships Analysis")

# Pair plot for numerical variables
pairplot_data = df[numerical_cols].dropna()
sns.pairplot(pairplot_data, diag_kind='kde', plot_kws={'alpha': 0.6})
plt.savefig('results/plots/multivariate_pairplot.png', dpi=300, bbox_inches='tight')
plt.close()

# 3.2 Multiple Variable Interactions
print("\n3.2. Multiple Variable Interactions")

# Salary by Gender and Senior Management
print("\nSalary by Gender and Senior Management:")
print(df.groupby(['Gender', 'Senior_Management'])['Salary'].agg(['mean', 'median', 'count']))

# Salary by Team and Senior Management (top 5 teams)
print("\nSalary by Team and Senior Management (Top 5 Teams):")
top_5_teams = df['Team'].value_counts().head(5).index
print(df[df['Team'].isin(top_5_teams)].groupby(['Team', 'Senior_Management'])['Salary'].agg(['mean', 'median', 'count']))

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Salary by Gender and Senior Management - Grouped bar chart
salary_by_gender_sm = df.groupby(['Gender', 'Senior_Management'])['Salary'].mean().unstack()
salary_by_gender_sm.plot(kind='bar', ax=axes[0, 0], color=['lightcoral', 'lightgreen'])
axes[0, 0].set_title('Average Salary by Gender and Senior Management')
axes[0, 0].set_xlabel('Gender')
axes[0, 0].set_ylabel('Average Salary')
axes[0, 0].legend(title='Senior Management')
axes[0, 0].tick_params(axis='x', rotation=0)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Salary by Team and Senior Management - Heatmap
salary_by_team_sm = df[df['Team'].isin(top_5_teams)].groupby(['Team', 'Senior_Management'])['Salary'].mean().unstack()
sns.heatmap(salary_by_team_sm, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[0, 1])
axes[0, 1].set_title('Average Salary by Team and Senior Management (Top 5 Teams)')
axes[0, 1].set_xlabel('Senior Management')
axes[0, 1].set_ylabel('Team')

# 3D scatter plot (Salary, Bonus %, Years of Service) colored by Gender
from mpl_toolkits.mplot3d import Axes3D
ax = fig.add_subplot(2, 2, 3, projection='3d')
for gender in df['Gender'].dropna().unique():
    gender_data = df[df['Gender'] == gender]
    ax.scatter(gender_data['Salary'], gender_data['Bonus_pct'], gender_data['Years_of_Service'],
               label=gender, alpha=0.6, s=50)
ax.set_xlabel('Salary')
ax.set_ylabel('Bonus %')
ax.set_zlabel('Years of Service')
ax.set_title('3D Scatter: Salary, Bonus %, Years of Service by Gender')
ax.legend()

# Correlation heatmap with all variables
# Create correlation matrix including encoded categorical variables
df_encoded = df.copy()
df_encoded['Gender_encoded'] = df_encoded['Gender'].map({'Male': 1, 'Female': 0}).fillna(0.5)
df_encoded['Senior_Management_encoded'] = df_encoded['Senior_Management'].astype(int)
correlation_all = df_encoded[['Salary', 'Bonus_pct', 'Years_of_Service', 
                               'Gender_encoded', 'Senior_Management_encoded']].corr()
sns.heatmap(correlation_all, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, ax=axes[1, 1], fmt='.3f')
axes[1, 1].set_title('Correlation Matrix (All Variables)')

plt.tight_layout()
plt.savefig('results/plots/multivariate_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 3.3 Advanced Multivariate Visualizations
print("\n3.3. Advanced Multivariate Visualizations")

# Faceted scatter plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Salary vs Bonus % by Gender
for gender in df['Gender'].dropna().unique():
    gender_data = df[df['Gender'] == gender]
    axes[0].scatter(gender_data['Salary'], gender_data['Bonus_pct'], 
                   label=gender, alpha=0.6, s=50)
axes[0].set_xlabel('Salary')
axes[0].set_ylabel('Bonus %')
axes[0].set_title('Salary vs Bonus % by Gender')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Salary vs Bonus % by Senior Management
for sm in [True, False]:
    sm_data = df[df['Senior_Management'] == sm]
    axes[1].scatter(sm_data['Salary'], sm_data['Bonus_pct'], 
                   label='Senior Management' if sm else 'Non-Senior Management', 
                   alpha=0.6, s=50)
axes[1].set_xlabel('Salary')
axes[1].set_ylabel('Bonus %')
axes[1].set_title('Salary vs Bonus % by Senior Management')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/multivariate_faceted.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*80)
print("UNIVARIATE, BIVARIATE, AND MULTIVARIATE ANALYSIS COMPLETED!")
print("="*80)
print(f"\nResults saved in:")
print("- Tables: results/tables/")
print("- Plots: results/plots/")
