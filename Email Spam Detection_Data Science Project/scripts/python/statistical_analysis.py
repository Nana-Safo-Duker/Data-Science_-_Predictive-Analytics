"""
Descriptive, Inferential, and Exploratory Statistical Analysis
Email Spam Detection Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_data():
    """Load processed dataset"""
    try:
        df = pd.read_csv('../../data/emails_spam_processed.csv')
    except:
        df = pd.read_csv('../../data/emails_spam_clean.csv')
        # Calculate basic features if not present
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
    return df

def descriptive_statistics(df):
    """Compute descriptive statistics"""
    print("="*60)
    print("DESCRIPTIVE STATISTICS")
    print("="*60)
    
    # Overall descriptive statistics
    numeric_cols = ['text_length', 'word_count', 'sentence_count', 'avg_word_length']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    print("\n1. Overall Descriptive Statistics:")
    print(df[numeric_cols].describe())
    
    # Statistics by class
    print("\n2. Descriptive Statistics by Class (Spam vs Ham):")
    for col in numeric_cols:
        print(f"\n{col}:")
        print(df.groupby('spam')[col].describe())
    
    # Central tendencies
    print("\n3. Central Tendency Measures:")
    for col in numeric_cols:
        print(f"\n{col}:")
        print(f"  Mean (Spam): {df[df['spam']==1][col].mean():.2f}")
        print(f"  Mean (Ham): {df[df['spam']==0][col].mean():.2f}")
        print(f"  Median (Spam): {df[df['spam']==1][col].median():.2f}")
        print(f"  Median (Ham): {df[df['spam']==0][col].median():.2f}")
        print(f"  Mode (Spam): {df[df['spam']==1][col].mode().iloc[0] if len(df[df['spam']==1][col].mode()) > 0 else 'N/A'}")
        print(f"  Mode (Ham): {df[df['spam']==0][col].mode().iloc[0] if len(df[df['spam']==0][col].mode()) > 0 else 'N/A'}")
    
    # Dispersion measures
    print("\n4. Dispersion Measures:")
    for col in numeric_cols:
        print(f"\n{col}:")
        print(f"  Std Dev (Spam): {df[df['spam']==1][col].std():.2f}")
        print(f"  Std Dev (Ham): {df[df['spam']==0][col].std():.2f}")
        print(f"  Variance (Spam): {df[df['spam']==1][col].var():.2f}")
        print(f"  Variance (Ham): {df[df['spam']==0][col].var():.2f}")
        print(f"  Range (Spam): {df[df['spam']==1][col].max() - df[df['spam']==1][col].min():.2f}")
        print(f"  Range (Ham): {df[df['spam']==0][col].max() - df[df['spam']==0][col].min():.2f}")
        print(f"  IQR (Spam): {df[df['spam']==1][col].quantile(0.75) - df[df['spam']==1][col].quantile(0.25):.2f}")
        print(f"  IQR (Ham): {df[df['spam']==0][col].quantile(0.75) - df[df['spam']==0][col].quantile(0.25):.2f}")
    
    # Skewness and Kurtosis
    print("\n5. Shape Measures (Skewness and Kurtosis):")
    for col in numeric_cols:
        print(f"\n{col}:")
        print(f"  Skewness (Spam): {df[df['spam']==1][col].skew():.2f}")
        print(f"  Skewness (Ham): {df[df['spam']==0][col].skew():.2f}")
        print(f"  Kurtosis (Spam): {df[df['spam']==1][col].kurtosis():.2f}")
        print(f"  Kurtosis (Ham): {df[df['spam']==0][col].kurtosis():.2f}")

def inferential_statistics(df):
    """Perform inferential statistical tests"""
    print("\n" + "="*60)
    print("INFERENTIAL STATISTICS")
    print("="*60)
    
    numeric_cols = ['text_length', 'word_count', 'sentence_count', 'avg_word_length']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    # T-test for each numeric variable
    print("\n1. Independent Samples T-Test:")
    print("Testing if there's a significant difference in means between Spam and Ham")
    print("\nHypotheses:")
    print("H0: μ_spam = μ_ham (no difference in means)")
    print("H1: μ_spam ≠ μ_ham (significant difference in means)")
    print("\nSignificance level: α = 0.05")
    
    for col in numeric_cols:
        spam_data = df[df['spam']==1][col].dropna()
        ham_data = df[df['spam']==0][col].dropna()
        
        # Perform t-test
        t_stat, p_value = ttest_ind(spam_data, ham_data)
        
        print(f"\n{col}:")
        print(f"  T-statistic: {t_stat:.4f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Mean (Spam): {spam_data.mean():.2f}")
        print(f"  Mean (Ham): {ham_data.mean():.2f}")
        
        if p_value < 0.05:
            print(f"  Result: Reject H0 - Significant difference (p < 0.05)")
        else:
            print(f"  Result: Fail to reject H0 - No significant difference (p >= 0.05)")
    
    # Mann-Whitney U test (non-parametric alternative)
    print("\n2. Mann-Whitney U Test (Non-parametric):")
    for col in numeric_cols:
        spam_data = df[df['spam']==1][col].dropna()
        ham_data = df[df['spam']==0][col].dropna()
        
        u_stat, p_value = mannwhitneyu(spam_data, ham_data, alternative='two-sided')
        
        print(f"\n{col}:")
        print(f"  U-statistic: {u_stat:.4f}")
        print(f"  P-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print(f"  Result: Reject H0 - Significant difference (p < 0.05)")
        else:
            print(f"  Result: Fail to reject H0 - No significant difference (p >= 0.05)")
    
    # Chi-square test for categorical associations
    print("\n3. Chi-Square Test for Independence:")
    # Create categorical variables for analysis
    df['text_length_category'] = pd.cut(df['text_length'], bins=3, labels=['Short', 'Medium', 'Long'])
    contingency_table = pd.crosstab(df['spam'], df['text_length_category'])
    
    print("\nContingency Table:")
    print(contingency_table)
    
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-square statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Degrees of freedom: {dof}")
    
    if p_value < 0.05:
        print("Result: Reject H0 - Text length category and spam are associated (p < 0.05)")
    else:
        print("Result: Fail to reject H0 - No association (p >= 0.05)")

def exploratory_statistical_analysis(df):
    """Perform exploratory statistical analysis"""
    print("\n" + "="*60)
    print("EXPLORATORY STATISTICAL ANALYSIS")
    print("="*60)
    
    numeric_cols = ['text_length', 'word_count', 'sentence_count', 'avg_word_length']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    # Correlation analysis
    print("\n1. Correlation Analysis:")
    correlation_matrix = df[numeric_cols + ['spam']].corr()
    print(correlation_matrix)
    
    # Visualize correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('../../output/figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Distribution analysis
    print("\n2. Distribution Analysis:")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for idx, col in enumerate(numeric_cols[:4]):
        # Histogram with KDE
        df[df['spam']==0][col].hist(ax=axes[idx], alpha=0.6, label='Ham', bins=30, color='skyblue')
        df[df['spam']==1][col].hist(ax=axes[idx], alpha=0.6, label='Spam', bins=30, color='salmon')
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../output/figures/distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Q-Q plots for normality check
    print("\n3. Normality Tests (Shapiro-Wilk):")
    for col in numeric_cols:
        spam_data = df[df['spam']==1][col].dropna().sample(min(5000, len(df[df['spam']==1][col])))
        ham_data = df[df['spam']==0][col].dropna().sample(min(5000, len(df[df['spam']==0][col])))
        
        stat_spam, p_spam = stats.shapiro(spam_data)
        stat_ham, p_ham = stats.shapiro(ham_data)
        
        print(f"\n{col}:")
        print(f"  Spam - Statistic: {stat_spam:.4f}, P-value: {p_spam:.6f}")
        print(f"    {'Normal' if p_spam > 0.05 else 'Not Normal'} distribution")
        print(f"  Ham - Statistic: {stat_ham:.4f}, P-value: {p_ham:.6f}")
        print(f"    {'Normal' if p_ham > 0.05 else 'Not Normal'} distribution")
    
    # Confidence intervals
    print("\n4. Confidence Intervals (95%):")
    for col in numeric_cols:
        spam_data = df[df['spam']==1][col].dropna()
        ham_data = df[df['spam']==0][col].dropna()
        
        spam_ci = stats.t.interval(0.95, len(spam_data)-1, 
                                   loc=spam_data.mean(), 
                                   scale=stats.sem(spam_data))
        ham_ci = stats.t.interval(0.95, len(ham_data)-1, 
                                  loc=ham_data.mean(), 
                                  scale=stats.sem(ham_data))
        
        print(f"\n{col}:")
        print(f"  Spam Mean: {spam_data.mean():.2f}, 95% CI: [{spam_ci[0]:.2f}, {spam_ci[1]:.2f}]")
        print(f"  Ham Mean: {ham_data.mean():.2f}, 95% CI: [{ham_ci[0]:.2f}, {ham_ci[1]:.2f}]")

def main():
    """Main function"""
    print("="*60)
    print("STATISTICAL ANALYSIS - Email Spam Detection")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Descriptive statistics
    descriptive_statistics(df)
    
    # Inferential statistics
    inferential_statistics(df)
    
    # Exploratory statistical analysis
    exploratory_statistical_analysis(df)
    
    print("\n" + "="*60)
    print("Statistical Analysis Complete!")
    print("="*60)

if __name__ == "__main__":
    main()


