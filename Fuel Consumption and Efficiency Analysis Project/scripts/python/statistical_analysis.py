"""
Statistical Analysis Script
Performs descriptive, inferential, and exploratory statistical analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import normaltest, shapiro, pearsonr, spearmanr
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_data():
    """Load the dataset"""
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'FuelConsumption.csv')
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    return df

def descriptive_statistics(df):
    """Calculate descriptive statistics"""
    print("="*50)
    print("DESCRIPTIVE STATISTICS")
    print("="*50)
    
    numerical_cols = ['ENGINE SIZE', 'CYLINDERS', 'FUEL CONSUMPTION', 'COEMISSIONS']
    
    descriptive_stats = pd.DataFrame({
        'Mean': df[numerical_cols].mean(),
        'Median': df[numerical_cols].median(),
        'Std': df[numerical_cols].std(),
        'Variance': df[numerical_cols].var(),
        'Min': df[numerical_cols].min(),
        'Max': df[numerical_cols].max(),
        'Range': df[numerical_cols].max() - df[numerical_cols].min(),
        'Skewness': df[numerical_cols].skew(),
        'Kurtosis': df[numerical_cols].kurtosis()
    })
    
    print("\nDescriptive Statistics:")
    print(descriptive_stats)
    
    print("\nQuartiles and IQR:")
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q2 = df[col].quantile(0.50)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        print(f"\n{col}:")
        print(f"  Q1: {Q1:.2f}, Q2: {Q2:.2f}, Q3: {Q3:.2f}")
        print(f"  IQR: {IQR:.2f}")

def inferential_statistics(df):
    """Perform inferential statistics"""
    print("\n" + "="*50)
    print("INFERENTIAL STATISTICS")
    print("="*50)
    
    numerical_cols = ['ENGINE SIZE', 'CYLINDERS', 'FUEL CONSUMPTION', 'COEMISSIONS']
    
    print("\nNormality Tests:")
    for col in numerical_cols:
        data = df[col].dropna()
        if len(data) < 5000:
            stat, p_value = shapiro(data)
            test_name = "Shapiro-Wilk"
        else:
            stat, p_value = normaltest(data)
            test_name = "D'Agostino"
        
        print(f"\n{col} ({test_name}):")
        print(f"  Statistic: {stat:.4f}, p-value: {p_value:.4f}")
        print(f"  Normal: {'Yes' if p_value > 0.05 else 'No'}")

def exploratory_statistics(df):
    """Perform exploratory statistical analysis"""
    print("\n" + "="*50)
    print("EXPLORATORY STATISTICAL ANALYSIS")
    print("="*50)
    
    numerical_cols = ['ENGINE SIZE', 'CYLINDERS', 'FUEL CONSUMPTION', 'COEMISSIONS']
    
    print("\n95% Confidence Intervals for Mean:")
    for col in numerical_cols:
        data = df[col].dropna()
        mean = data.mean()
        n = len(data)
        ci = stats.t.interval(0.95, n-1, loc=mean, scale=stats.sem(data))
        print(f"\n{col}:")
        print(f"  Mean: {mean:.2f}")
        print(f"  95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
    
    print("\nCorrelation Analysis with Significance:")
    target = 'COEMISSIONS'
    for col in ['ENGINE SIZE', 'CYLINDERS', 'FUEL CONSUMPTION']:
        data = df[[col, target]].dropna()
        pearson_corr, pearson_p = pearsonr(data[col], data[target])
        spearman_corr, spearman_p = spearmanr(data[col], data[target])
        print(f"\n{col} vs {target}:")
        print(f"  Pearson: r={pearson_corr:.4f}, p={pearson_p:.4f}")
        print(f"  Spearman: ρ={spearman_corr:.4f}, p={spearman_p:.4f}")

def main():
    """Main function"""
    df = load_data()
    descriptive_statistics(df)
    inferential_statistics(df)
    exploratory_statistics(df)

if __name__ == "__main__":
    main()


