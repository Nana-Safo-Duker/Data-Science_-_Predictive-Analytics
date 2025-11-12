"""
Univariate, Bivariate, and Multivariate Analysis Script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import stats

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
np.random.seed(42)

def univariate_analysis(df, output_dir):
    """Perform univariate analysis."""
    print("Univariate Analysis")
    print("="*80)
    
    if 'TransactionAmt' in df.columns:
        print(f"\nTransactionAmt Statistics:")
        print(f"Mean: {df['TransactionAmt'].mean():.2f}")
        print(f"Median: {df['TransactionAmt'].median():.2f}")
        print(f"Std: {df['TransactionAmt'].std():.2f}")
        print(f"Skewness: {df['TransactionAmt'].skew():.4f}")
        print(f"Kurtosis: {df['TransactionAmt'].kurtosis():.4f}")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist(df['TransactionAmt'], bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_title('Histogram')
        axes[1].boxplot(df['TransactionAmt'])
        axes[1].set_title('Box Plot')
        stats.probplot(df['TransactionAmt'], dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q Plot')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/univariate_transactionamt.png', dpi=300, bbox_inches='tight')
        plt.close()

def bivariate_analysis(df, output_dir):
    """Perform bivariate analysis."""
    print("\nBivariate Analysis")
    print("="*80)
    
    if 'TransactionAmt' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        df.boxplot(column='TransactionAmt', by='isFraud', ax=axes[0])
        axes[0].set_title('Transaction Amount by Fraud Status')
        sns.violinplot(data=df, x='isFraud', y='TransactionAmt', ax=axes[1])
        axes[1].set_title('Distribution by Fraud Status')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/bivariate_transaction_fraud.png', dpi=300, bbox_inches='tight')
        plt.close()

def multivariate_analysis(df, output_dir):
    """Perform multivariate analysis."""
    print("\nMultivariate Analysis")
    print("="*80)
    
    key_features = ['TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'isFraud']
    key_features = [f for f in key_features if f in df.columns]
    
    if len(key_features) > 1:
        corr_matrix = df[key_features].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0)
        plt.title('Multivariate Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/multivariate_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Correlation matrix saved!")

def main():
    """Main function."""
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'data' / 'fraud_data.csv'
    output_dir = project_root / 'outputs' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(data_path)
    print(f"Data loaded: {df.shape}")
    
    univariate_analysis(df, output_dir)
    bivariate_analysis(df, output_dir)
    multivariate_analysis(df, output_dir)
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)

if __name__ == "__main__":
    main()

