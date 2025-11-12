"""
Statistical Analysis Script for Fraud Detection Dataset

This script performs comprehensive statistical analysis including:
1. Descriptive Statistics - Mean, median, mode, standard deviation, variance, skewness, kurtosis
2. Inferential Statistics - Hypothesis testing, confidence intervals, t-tests, chi-square tests
3. Exploratory Statistics - Correlation analysis, feature relationships, statistical tests
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest, shapiro, chi2_contingency, ttest_ind, mannwhitneyu
import warnings
from pathlib import Path
from statsmodels.stats.proportion import proportions_ztest

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
np.random.seed(42)

def descriptive_statistics(df, output_dir):
    """Perform descriptive statistical analysis."""
    print("\n" + "="*80)
    print("DESCRIPTIVE STATISTICS")
    print("="*80)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    key_features = ['TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2']
    key_features = [f for f in key_features if f in numerical_cols]
    
    print("\nDescriptive Statistics for Key Numerical Features:")
    desc_stats = df[key_features].describe()
    print(desc_stats)
    
    # Additional statistics: skewness and kurtosis
    print("\nSkewness and Kurtosis:")
    skew_kurt = pd.DataFrame({
        'Skewness': df[key_features].skew(),
        'Kurtosis': df[key_features].kurtosis()
    })
    print(skew_kurt)
    
    # Descriptive statistics by fraud status
    if 'TransactionAmt' in df.columns:
        print("\nTransaction Amount Statistics by Fraud Status:")
        fraud_stats = df.groupby('isFraud')['TransactionAmt'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max', 
            'skew', pd.Series.kurt
        ]).round(4)
        fraud_stats.columns = ['Count', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Skewness', 'Kurtosis']
        print(fraud_stats)
        
        # Visualize distributions
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        df[df['isFraud']==0]['TransactionAmt'].hist(bins=50, alpha=0.7, label='Legitimate', ax=axes[0], color='blue')
        df[df['isFraud']==1]['TransactionAmt'].hist(bins=50, alpha=0.7, label='Fraud', ax=axes[0], color='red')
        axes[0].set_xlabel('Transaction Amount')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Transaction Amount Distribution by Fraud Status')
        axes[0].legend()
        axes[0].set_xlim(0, df['TransactionAmt'].quantile(0.99))
        
        # Box plot
        df.boxplot(column='TransactionAmt', by='isFraud', ax=axes[1])
        axes[1].set_xlabel('Fraud Status')
        axes[1].set_ylabel('Transaction Amount')
        axes[1].set_title('Transaction Amount by Fraud Status')
        axes[1].set_yscale('log')
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/descriptive_stats_transaction.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nFigure saved to {output_dir}/descriptive_stats_transaction.png")

def inferential_statistics(df, output_dir):
    """Perform inferential statistical analysis."""
    print("\n" + "="*80)
    print("INFERENTIAL STATISTICS")
    print("="*80)
    
    # Hypothesis Test 1: Test if mean transaction amount differs between fraud and legitimate transactions
    if 'TransactionAmt' in df.columns:
        print("\nHypothesis Test: Transaction Amount")
        print("H0: Mean transaction amount is the same for fraud and legitimate transactions")
        print("H1: Mean transaction amount differs between fraud and legitimate transactions")
        print("\n")
        
        fraud_amt = df[df['isFraud']==1]['TransactionAmt'].dropna()
        legit_amt = df[df['isFraud']==0]['TransactionAmt'].dropna()
        
        # Check normality (Shapiro-Wilk test on sample)
        sample_size = min(5000, len(fraud_amt), len(legit_amt))
        _, p_fraud_norm = shapiro(fraud_amt.sample(sample_size, random_state=42))
        _, p_legit_norm = shapiro(legit_amt.sample(sample_size, random_state=42))
        
        print(f"Normality test p-values:")
        print(f"  Fraud transactions: {p_fraud_norm:.6f}")
        print(f"  Legitimate transactions: {p_legit_norm:.6f}")
        print(f"  (Both distributions are likely non-normal, p < 0.05)")
        
        # Use Mann-Whitney U test (non-parametric)
        statistic, p_value = mannwhitneyu(fraud_amt, legit_amt, alternative='two-sided')
        print(f"\nMann-Whitney U Test Results:")
        print(f"  Statistic: {statistic:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Significance level: 0.05")
        
        if p_value < 0.05:
            print(f"  Result: REJECT H0 - Mean transaction amounts are significantly different")
        else:
            print(f"  Result: FAIL TO REJECT H0 - No significant difference in mean transaction amounts")
        
        # Also perform t-test for comparison
        t_stat, t_pvalue = ttest_ind(fraud_amt, legit_amt)
        print(f"\nIndependent t-test Results (for comparison):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {t_pvalue:.6f}")
        
        # Calculate confidence intervals
        fraud_mean = fraud_amt.mean()
        fraud_std = fraud_amt.std()
        fraud_se = fraud_std / np.sqrt(len(fraud_amt))
        fraud_ci = stats.norm.interval(0.95, loc=fraud_mean, scale=fraud_se)
        
        legit_mean = legit_amt.mean()
        legit_std = legit_amt.std()
        legit_se = legit_std / np.sqrt(len(legit_amt))
        legit_ci = stats.norm.interval(0.95, loc=legit_mean, scale=legit_se)
        
        print(f"\n95% Confidence Intervals:")
        print(f"  Fraud transactions: ${fraud_ci[0]:.2f} - ${fraud_ci[1]:.2f} (mean: ${fraud_mean:.2f})")
        print(f"  Legitimate transactions: ${legit_ci[0]:.2f} - ${legit_ci[1]:.2f} (mean: ${legit_mean:.2f})")
    
    # Hypothesis Test 2: Chi-square test for categorical variables
    print("\n" + "="*80)
    print("Chi-square Test: ProductCD and Fraud")
    print("="*80)
    
    if 'ProductCD' in df.columns:
        # Create contingency table
        contingency_table = pd.crosstab(df['ProductCD'], df['isFraud'])
        print("\nContingency Table:")
        print(contingency_table)
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        print(f"\nChi-square Test Results:")
        print(f"  Chi-square statistic: {chi2:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Degrees of freedom: {dof}")
        print(f"  Significance level: 0.05")
        
        if p_value < 0.05:
            print(f"  Result: REJECT H0 - ProductCD is significantly associated with fraud")
        else:
            print(f"  Result: FAIL TO REJECT H0 - No significant association")
        
        # Visualize
        contingency_pct = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
        contingency_pct.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.title('Fraud Rate by ProductCD', fontsize=14, fontweight='bold')
        plt.xlabel('ProductCD')
        plt.ylabel('Percentage (%)')
        plt.legend(['Legitimate', 'Fraud'])
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/chi_square_productcd.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nFigure saved to {output_dir}/chi_square_productcd.png")
    
    # Hypothesis Test 3: Test for card type and fraud
    if 'card4' in df.columns:
        print("\n" + "="*80)
        print("Chi-square Test: Card Type (card4) and Fraud")
        print("="*80)
        
        contingency_table = pd.crosstab(df['card4'], df['isFraud'])
        print("\nContingency Table:")
        print(contingency_table)
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        print(f"\nChi-square Test Results:")
        print(f"  Chi-square statistic: {chi2:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Degrees of freedom: {dof}")
        
        if p_value < 0.05:
            print(f"  Result: REJECT H0 - Card type is significantly associated with fraud")
        else:
            print(f"  Result: FAIL TO REJECT H0 - No significant association")

def exploratory_statistics(df, output_dir):
    """Perform exploratory statistical analysis."""
    print("\n" + "="*80)
    print("EXPLORATORY STATISTICS")
    print("="*80)
    
    # Correlation analysis
    print("\nCorrelation Analysis")
    key_features = ['TransactionAmt', 'card1', 'card2', 'card3', 'card5', 
                    'addr1', 'addr2', 'dist1', 'dist2', 'isFraud']
    key_features = [f for f in key_features if f in df.columns and df[f].dtype in [np.int64, np.float64]]
    
    if len(key_features) > 1:
        # Calculate correlation matrix
        corr_matrix = df[key_features].corr()
        
        # Visualize correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Key Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_matrix_statistical.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nFigure saved to {output_dir}/correlation_matrix_statistical.png")
        
        # Features most correlated with fraud
        if 'isFraud' in corr_matrix.columns:
            fraud_corr = corr_matrix['isFraud'].sort_values(ascending=False)
            print("\nFeatures most correlated with Fraud:")
            print(fraud_corr)
            
            # Statistical significance of correlations
            print("\nStatistical Significance of Correlations with Fraud:")
            n = len(df)
            significant_features = []
            for feature in fraud_corr.index:
                if feature != 'isFraud':
                    corr_coef = fraud_corr[feature]
                    # Test significance of correlation
                    t_stat = corr_coef * np.sqrt((n - 2) / (1 - corr_coef**2))
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                    if p_value < 0.05:
                        significant_features.append((feature, corr_coef, p_value))
            
            if significant_features:
                sig_df = pd.DataFrame(significant_features, columns=['Feature', 'Correlation', 'p-value'])
                print("\nSignificantly correlated features (p < 0.05):")
                print(sig_df)
    
    # Statistical summary by fraud status for multiple features
    print("\n" + "="*80)
    print("Statistical Summary by Fraud Status")
    print("="*80)
    
    analysis_features = ['TransactionAmt', 'card1', 'card2', 'card3', 'card5']
    analysis_features = [f for f in analysis_features if f in df.columns]
    
    if analysis_features:
        summary_stats = df.groupby('isFraud')[analysis_features].agg(['mean', 'median', 'std', 'skew'])
        print("\nSummary Statistics by Fraud Status:")
        print(summary_stats)
        
        # Perform statistical tests for each feature
        print("\nStatistical Tests for Each Feature:")
        test_results = []
        
        for feature in analysis_features:
            fraud_data = df[df['isFraud']==1][feature].dropna()
            legit_data = df[df['isFraud']==0][feature].dropna()
            
            if len(fraud_data) > 0 and len(legit_data) > 0:
                # Mann-Whitney U test
                statistic, p_value = mannwhitneyu(fraud_data, legit_data, alternative='two-sided')
                test_results.append({
                    'Feature': feature,
                    'Test': 'Mann-Whitney U',
                    'Statistic': statistic,
                    'p-value': p_value,
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                })
        
        test_df = pd.DataFrame(test_results)
        print(test_df)

def main():
    """Main function to run statistical analysis."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'data' / 'fraud_data.csv'
    output_dir = project_root / 'outputs' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"Data loaded: {df.shape}")
    print(f"Target variable distribution:\n{df['isFraud'].value_counts()}")
    
    # Perform analyses
    descriptive_statistics(df, output_dir)
    inferential_statistics(df, output_dir)
    exploratory_statistics(df, output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("="*80)
    print("\n1. Descriptive Statistics:")
    print("   - Calculated mean, median, std, skewness, and kurtosis for key features")
    print("   - Compared statistics between fraud and legitimate transactions")
    print("\n2. Inferential Statistics:")
    print("   - Performed hypothesis tests to compare fraud and legitimate transactions")
    print("   - Used non-parametric tests (Mann-Whitney U) due to non-normal distributions")
    print("   - Calculated confidence intervals for key metrics")
    print("   - Performed chi-square tests for categorical variables")
    print("\n3. Exploratory Statistics:")
    print("   - Analyzed correlations between features and fraud")
    print("   - Identified statistically significant relationships")
    print("   - Explored feature distributions by fraud status")
    print("\nStatistical Analysis Complete! Check outputs/figures/ for visualizations.")

if __name__ == "__main__":
    main()

