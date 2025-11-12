"""
Exploratory Data Analysis (EDA) for Employee Dataset
Comprehensive data exploration, cleaning, and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_root)

# Create results directories if they don't exist
os.makedirs('results/plots', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

print("="*80)
print("EXPLORATORY DATA ANALYSIS - EMPLOYEE DATASET")
print("="*80)

# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("\n1. Loading Data...")
df = pd.read_csv('data/raw/employees.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# =============================================================================
# 2. DATA OVERVIEW
# =============================================================================
print("\n2. Data Overview...")
print("\nFirst few rows:")
print(df.head(10))
print("\nData types:")
print(df.dtypes)
print("\nBasic information:")
print(df.info())

# =============================================================================
# 3. MISSING VALUES ANALYSIS
# =============================================================================
print("\n3. Missing Values Analysis...")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_data,
    'Percentage': missing_percent
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
print(missing_df)

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.tight_layout()
plt.savefig('results/plots/missing_values_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 4. DATA CLEANING AND PREPROCESSING
# =============================================================================
print("\n4. Data Cleaning and Preprocessing...")
df_clean = df.copy()

# Rename columns for easier handling
df_clean.columns = df_clean.columns.str.strip().str.replace(' ', '_').str.replace('%', 'pct')

# Handle missing values in First Name
df_clean['First_Name'] = df_clean['First_Name'].fillna('Unknown')

# Handle missing values in Gender
print(f"Gender distribution before cleaning: {df_clean['Gender'].value_counts()}")

# Handle missing values in Senior Management (convert to boolean)
df_clean['Senior_Management'] = df_clean['Senior_Management'].map({'true': True, 'false': False, True: True, False: False})
df_clean['Senior_Management'] = df_clean['Senior_Management'].fillna(False)

# Handle missing values in Team
df_clean['Team'] = df_clean['Team'].fillna('Unknown')

# Parse Start Date
df_clean['Start_Date'] = pd.to_datetime(df_clean['Start_Date'], errors='coerce', format='%m/%d/%Y')
df_clean['Start_Year'] = df_clean['Start_Date'].dt.year
df_clean['Start_Month'] = df_clean['Start_Date'].dt.month
df_clean['Years_of_Service'] = (datetime.now() - df_clean['Start_Date']).dt.days / 365.25

# Parse Last Login Time (extract hour)
df_clean['Last_Login_Time'] = pd.to_datetime(df_clean['Last_Login_Time'], errors='coerce', format='%I:%M %p')
df_clean['Last_Login_Hour'] = df_clean['Last_Login_Time'].dt.hour

# Ensure numeric columns are numeric
df_clean['Salary'] = pd.to_numeric(df_clean['Salary'], errors='coerce')
df_clean['Bonus_pct'] = pd.to_numeric(df_clean['Bonus_pct'], errors='coerce')

# Remove rows with missing critical data (Salary)
df_clean = df_clean.dropna(subset=['Salary'])

print(f"\nDataset shape after cleaning: {df_clean.shape}")
print(f"Rows removed: {len(df) - len(df_clean)}")

# Save cleaned dataset
df_clean.to_csv('data/processed/employees_cleaned.csv', index=False)
print("\nCleaned dataset saved to: data/processed/employees_cleaned.csv")

# =============================================================================
# 5. NUMERICAL VARIABLE ANALYSIS
# =============================================================================
print("\n5. Numerical Variable Analysis...")
numerical_cols = ['Salary', 'Bonus_pct', 'Years_of_Service']
print("\nDescriptive Statistics:")
print(df_clean[numerical_cols].describe())

# Distribution plots for numerical variables
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for i, col in enumerate(numerical_cols):
    # Histogram
    axes[i*2].hist(df_clean[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[i*2].set_title(f'Distribution of {col}')
    axes[i*2].set_xlabel(col)
    axes[i*2].set_ylabel('Frequency')
    axes[i*2].grid(True, alpha=0.3)
    
    # Box plot
    axes[i*2+1].boxplot(df_clean[col].dropna())
    axes[i*2+1].set_title(f'Box Plot of {col}')
    axes[i*2+1].set_ylabel(col)
    axes[i*2+1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/numerical_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 6. CATEGORICAL VARIABLE ANALYSIS
# =============================================================================
print("\n6. Categorical Variable Analysis...")
categorical_cols = ['Gender', 'Senior_Management', 'Team']

for col in categorical_cols:
    print(f"\n{col} distribution:")
    print(df_clean[col].value_counts())
    print(f"Unique values: {df_clean[col].nunique()}")

# Visualize categorical variables
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Gender distribution
gender_counts = df_clean['Gender'].value_counts()
axes[0].bar(gender_counts.index, gender_counts.values, color=['skyblue', 'pink', 'lightgray'])
axes[0].set_title('Gender Distribution')
axes[0].set_xlabel('Gender')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

# Senior Management distribution
sm_counts = df_clean['Senior_Management'].value_counts()
axes[1].bar(['False', 'True'], sm_counts.values, color=['lightcoral', 'lightgreen'])
axes[1].set_title('Senior Management Distribution')
axes[1].set_xlabel('Senior Management')
axes[1].set_ylabel('Count')

# Team distribution (top 10)
team_counts = df_clean['Team'].value_counts().head(10)
axes[2].barh(team_counts.index, team_counts.values, color='steelblue')
axes[2].set_title('Top 10 Teams by Employee Count')
axes[2].set_xlabel('Count')
axes[2].set_ylabel('Team')

plt.tight_layout()
plt.savefig('results/plots/categorical_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 7. OUTLIER DETECTION
# =============================================================================
print("\n7. Outlier Detection...")
for col in numerical_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
    print(f"\n{col} outliers: {len(outliers)} ({len(outliers)/len(df_clean)*100:.2f}%)")
    print(f"  Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")

# Visualize outliers
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(numerical_cols):
    axes[i].boxplot(df_clean[col].dropna(), vert=True)
    axes[i].set_title(f'Outliers in {col}')
    axes[i].set_ylabel(col)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/outliers_detection.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 8. CORRELATION ANALYSIS
# =============================================================================
print("\n8. Correlation Analysis...")
correlation_matrix = df_clean[numerical_cols].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Numerical Variables')
plt.tight_layout()
plt.savefig('results/plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 9. RELATIONSHIP ANALYSIS
# =============================================================================
print("\n9. Relationship Analysis...")

# Salary by Gender
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Salary distribution by Gender
df_clean.boxplot(column='Salary', by='Gender', ax=axes[0, 0])
axes[0, 0].set_title('Salary Distribution by Gender')
axes[0, 0].set_xlabel('Gender')
axes[0, 0].set_ylabel('Salary')
axes[0, 0].grid(True, alpha=0.3)

# Salary distribution by Senior Management
df_clean.boxplot(column='Salary', by='Senior_Management', ax=axes[0, 1])
axes[0, 1].set_title('Salary Distribution by Senior Management')
axes[0, 1].set_xlabel('Senior Management')
axes[0, 1].set_ylabel('Salary')
axes[0, 1].grid(True, alpha=0.3)

# Salary by Team (top 10 teams)
top_teams = df_clean['Team'].value_counts().head(10).index
df_top_teams = df_clean[df_clean['Team'].isin(top_teams)]
df_top_teams.boxplot(column='Salary', by='Team', ax=axes[1, 0])
axes[1, 0].set_title('Salary Distribution by Team (Top 10)')
axes[1, 0].set_xlabel('Team')
axes[1, 0].set_ylabel('Salary')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3)

# Salary vs Bonus %
axes[1, 1].scatter(df_clean['Salary'], df_clean['Bonus_pct'], alpha=0.5)
axes[1, 1].set_title('Salary vs Bonus %')
axes[1, 1].set_xlabel('Salary')
axes[1, 1].set_ylabel('Bonus %')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/relationship_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 10. TIME SERIES ANALYSIS (HIRING TRENDS)
# =============================================================================
print("\n10. Time Series Analysis (Hiring Trends)...")

# Hiring trends by year
hiring_by_year = df_clean.groupby('Start_Year').size().reset_index(name='Count')
hiring_by_year = hiring_by_year[hiring_by_year['Start_Year'].notna()]

plt.figure(figsize=(14, 6))
plt.plot(hiring_by_year['Start_Year'], hiring_by_year['Count'], marker='o', linewidth=2, markersize=8)
plt.title('Hiring Trends Over Years')
plt.xlabel('Year')
plt.ylabel('Number of Employees Hired')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/hiring_trends.png', dpi=300, bbox_inches='tight')
plt.close()

# Hiring by month
hiring_by_month = df_clean.groupby('Start_Month').size().reset_index(name='Count')
hiring_by_month = hiring_by_month[hiring_by_month['Start_Month'].notna()]

plt.figure(figsize=(12, 6))
plt.bar(hiring_by_month['Start_Month'], hiring_by_month['Count'], color='steelblue', edgecolor='black')
plt.title('Hiring Trends by Month')
plt.xlabel('Month')
plt.ylabel('Number of Employees Hired')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('results/plots/hiring_by_month.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 11. SUMMARY STATISTICS
# =============================================================================
print("\n11. Summary Statistics...")
print("\nOverall Statistics:")
print(f"Total employees: {len(df_clean)}")
print(f"Average salary: ${df_clean['Salary'].mean():,.2f}")
print(f"Median salary: ${df_clean['Salary'].median():,.2f}")
print(f"Average bonus %: {df_clean['Bonus_pct'].mean():.2f}%")
print(f"Average years of service: {df_clean['Years_of_Service'].mean():.2f} years")
print(f"Senior management percentage: {df_clean['Senior_Management'].sum() / len(df_clean) * 100:.2f}%")

# Gender statistics
print("\nGender Statistics:")
print(df_clean.groupby('Gender')['Salary'].agg(['mean', 'median', 'count']))

# Team statistics
print("\nTop 5 Teams by Average Salary:")
print(df_clean.groupby('Team')['Salary'].mean().sort_values(ascending=False).head(5))

print("\n" + "="*80)
print("EDA COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\nResults saved in:")
print("- Cleaned dataset: data/processed/employees_cleaned.csv")
print("- Plots: results/plots/")
