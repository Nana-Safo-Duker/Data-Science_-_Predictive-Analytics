"""
Statistical Analysis Script
Descriptive, Inferential, and Exploratory Statistical Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, normaltest
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Set paths
data_path = Path('../../data/Customers.csv')
results_path = Path('../../results')
results_path.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(data_path)

def descriptive_statistics():
    """Calculate descriptive statistics"""
    print("=" * 50)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 50)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print("\n=== Descriptive Statistics for Numerical Variables ===")
    print(df[numerical_cols].describe())
    
    # Additional descriptive statistics
    print("\n=== Additional Descriptive Statistics ===")
    desc_stats = pd.DataFrame({
        'Mean': df[numerical_cols].mean(),
        'Median': df[numerical_cols].median(),
        'Std Dev': df[numerical_cols].std(),
        'Variance': df[numerical_cols].var(),
        'Min': df[numerical_cols].min(),
        'Max': df[numerical_cols].max(),
        'Range': df[numerical_cols].max() - df[numerical_cols].min(),
        'Q1': df[numerical_cols].quantile(0.25),
        'Q3': df[numerical_cols].quantile(0.75),
        'IQR': df[numerical_cols].quantile(0.75) - df[numerical_cols].quantile(0.25),
        'Skewness': df[numerical_cols].skew(),
        'Kurtosis': df[numerical_cols].kurtosis()
    })
    print(desc_stats.T)
    
    # Categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print("\n=== Descriptive Statistics for Categorical Variables ===")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(f"  Count: {df[col].count()}")
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Mode: {df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A'}")
        print(f"  Mode frequency: {df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 'N/A'}")

def inferential_statistics():
    """Perform inferential statistics"""
    print("\n" + "=" * 50)
    print("INFERENTIAL STATISTICS")
    print("=" * 50)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Confidence intervals
    print("\n=== Confidence Intervals (95%) ===")
    for col in numerical_cols:
        mean = df[col].mean()
        std = df[col].std()
        n = len(df[col])
        
        margin_error = 1.96 * (std / np.sqrt(n))
        ci_lower = mean - margin_error
        ci_upper = mean + margin_error
        
        print(f"\n{col}:")
        print(f"  Mean: {mean:.2f}")
        print(f"  95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
    
    # Normality tests
    print("\n=== Normality Tests ===")
    alpha = 0.05
    for col in numerical_cols:
        data = df[col].dropna()
        stat, p_value = normaltest(data)
        
        print(f"\n{col}:")
        print(f"  D'Agostino-Pearson test:")
        print(f"    Statistic: {stat:.4f}")
        print(f"    p-value: {p_value:.4f}")
        
        if p_value > alpha:
            print(f"    Result: Data appears to be normally distributed (p > {alpha})")
        else:
            print(f"    Result: Data does not appear to be normally distributed (p <= {alpha})")
    
    # Chi-square test
    print("\n=== Chi-Square Test for Independence ===")
    contingency_table = pd.crosstab(df['Country'], df['City'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    print(f"\nChi-square statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"p-value: {p_value:.4f}")
    
    alpha = 0.05
    if p_value < alpha:
        print(f"\nResult: Reject null hypothesis. Country and City are not independent (p < {alpha})")
    else:
        print(f"\nResult: Fail to reject null hypothesis. Country and City may be independent (p >= {alpha})")

def exploratory_statistics():
    """Perform exploratory statistical analysis"""
    print("\n" + "=" * 50)
    print("EXPLORATORY STATISTICAL ANALYSIS")
    print("=" * 50)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Distribution analysis
    print("\n=== Distribution Analysis ===")
    for col in numerical_cols:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(df[col], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Distribution of {col}')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(df[col], vert=True)
        axes[1].set_ylabel(col)
        axes[1].set_title(f'Box Plot of {col}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_path / f'{col}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Outlier detection
    print("\n=== Outlier Detection (IQR Method) ===")
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        
        print(f"\n{col}:")
        print(f"  Lower bound: {lower_bound:.2f}")
        print(f"  Upper bound: {upper_bound:.2f}")
        print(f"  Number of outliers: {len(outliers)}")
        if len(outliers) > 0:
            print(f"  Outlier values: {outliers.tolist()}")
    
    # Central tendency and dispersion
    print("\n=== Central Tendency and Dispersion Measures ===")
    measures = pd.DataFrame({
        'Mean': df[numerical_cols].mean(),
        'Median': df[numerical_cols].median(),
        'Std Dev': df[numerical_cols].std(),
        'Variance': df[numerical_cols].var(),
        'CV (%)': (df[numerical_cols].std() / df[numerical_cols].mean() * 100)
    })
    print(measures)
    
    print("\n✓ Statistical analysis visualizations saved to results/ directory")

def main():
    """Main function"""
    descriptive_statistics()
    inferential_statistics()
    exploratory_statistics()
    
    print("\n" + "=" * 50)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("=" * 50)
    print("\n1. Descriptive Statistics:")
    print("   • Calculated measures of central tendency (mean, median, mode)")
    print("   • Calculated measures of dispersion (std dev, variance, IQR)")
    print("   • Analyzed distribution characteristics (skewness, kurtosis)")
    print("\n2. Inferential Statistics:")
    print("   • Calculated 95% confidence intervals")
    print("   • Performed normality tests")
    print("   • Conducted chi-square tests for independence")
    print("\n3. Exploratory Statistics:")
    print("   • Analyzed distributions")
    print("   • Detected outliers")
    print("   • Examined central tendency and dispersion")

if __name__ == "__main__":
    main()

