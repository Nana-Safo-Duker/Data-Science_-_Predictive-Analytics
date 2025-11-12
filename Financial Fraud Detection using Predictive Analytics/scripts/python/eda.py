"""
Exploratory Data Analysis Script for Fraud Detection Dataset

This script performs comprehensive EDA on the fraud detection dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Set style
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Set random seed
np.random.seed(42)

def load_data(data_path):
    """Load the dataset."""
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    return df

def analyze_target_variable(df, output_dir):
    """Analyze the target variable distribution."""
    print("\n" + "="*80)
    print("Target Variable Analysis")
    print("="*80)
    
    target_counts = df['isFraud'].value_counts()
    target_percentages = df['isFraud'].value_counts(normalize=True) * 100
    
    print(f"\nCounts:\n{target_counts}")
    print(f"\nPercentages:\n{target_percentages}")
    print(f"\nFraud Rate: {df['isFraud'].mean():.4f} ({df['isFraud'].mean()*100:.2f}%)")
    print(f"Class Imbalance Ratio: {target_counts[0] / target_counts[1]:.2f}:1")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(target_counts.index, target_counts.values, color=['#3498db', '#e74c3c'])
    axes[0].set_xlabel('isFraud', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Fraud Distribution (Count)', fontsize=14, fontweight='bold')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Legitimate (0)', 'Fraud (1)'])
    
    axes[1].bar(target_percentages.index, target_percentages.values, color=['#3498db', '#e74c3c'])
    axes[1].set_xlabel('isFraud', fontsize=12)
    axes[1].set_ylabel('Percentage (%)', fontsize=12)
    axes[1].set_title('Fraud Distribution (Percentage)', fontsize=14, fontweight='bold')
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Legitimate (0)', 'Fraud (1)'])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to {output_dir}/target_distribution.png")

def analyze_missing_values(df, output_dir):
    """Analyze missing values in the dataset."""
    print("\n" + "="*80)
    print("Missing Values Analysis")
    print("="*80)
    
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
        'Data_Type': df.dtypes
    })
    
    missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
    
    print(f"\nTotal columns with missing values: {len(missing_data)}")
    print(f"Total missing values: {df.isnull().sum().sum()}")
    print(f"Percentage of missing data: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.2f}%")
    
    if len(missing_data) > 0:
        print("\nTop 20 columns with most missing values:")
        print(missing_data.head(20).to_string())
        
        # Visualize
        plt.figure(figsize=(12, 8))
        top_missing = missing_data.head(30)
        plt.barh(range(len(top_missing)), top_missing['Missing_Percentage'].values)
        plt.yticks(range(len(top_missing)), top_missing['Column'].values)
        plt.xlabel('Missing Percentage (%)', fontsize=12)
        plt.ylabel('Columns', fontsize=12)
        plt.title('Top 30 Columns with Missing Values', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/missing_values.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nFigure saved to {output_dir}/missing_values.png")
    else:
        print("No missing values found in the dataset!")

def analyze_transaction_amount(df, output_dir):
    """Analyze transaction amount distribution."""
    if 'TransactionAmt' not in df.columns:
        print("TransactionAmt column not found. Skipping transaction amount analysis.")
        return
    
    print("\n" + "="*80)
    print("Transaction Amount Analysis")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Distribution
    axes[0, 0].hist(df['TransactionAmt'], bins=50, color='#3498db', edgecolor='black')
    axes[0, 0].set_xlabel('Transaction Amount', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Transaction Amount Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlim(0, df['TransactionAmt'].quantile(0.99))
    
    # Log distribution
    log_trans_amt = np.log1p(df['TransactionAmt'])
    axes[0, 1].hist(log_trans_amt, bins=50, color='#2ecc71', edgecolor='black')
    axes[0, 1].set_xlabel('Log(Transaction Amount + 1)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Log Transaction Amount Distribution', fontsize=14, fontweight='bold')
    
    # Box plot by fraud status
    fraud_data = [df[df['isFraud']==0]['TransactionAmt'], df[df['isFraud']==1]['TransactionAmt']]
    axes[1, 0].boxplot(fraud_data, labels=['Legitimate', 'Fraud'])
    axes[1, 0].set_ylabel('Transaction Amount', fontsize=12)
    axes[1, 0].set_title('Transaction Amount by Fraud Status', fontsize=14, fontweight='bold')
    axes[1, 0].set_yscale('log')
    
    # Violin plot
    df_temp = df[df['TransactionAmt'] <= df['TransactionAmt'].quantile(0.95)]
    sns.violinplot(data=df_temp, x='isFraud', y='TransactionAmt', ax=axes[1, 1])
    axes[1, 1].set_xlabel('isFraud', fontsize=12)
    axes[1, 1].set_ylabel('Transaction Amount', fontsize=12)
    axes[1, 1].set_title('Transaction Amount Distribution by Fraud', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticklabels(['Legitimate', 'Fraud'])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/transaction_amount_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to {output_dir}/transaction_amount_analysis.png")
    
    # Statistical summary
    print("\nTransaction Amount Statistics by Fraud Status:")
    print(df.groupby('isFraud')['TransactionAmt'].describe())

def main():
    """Main function to run EDA."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'data' / 'fraud_data.csv'
    output_dir = project_root / 'outputs' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data(data_path)
    
    # Perform analyses
    analyze_target_variable(df, output_dir)
    analyze_missing_values(df, output_dir)
    analyze_transaction_amount(df, output_dir)
    
    # Basic summary
    print("\n" + "="*80)
    print("EDA SUMMARY")
    print("="*80)
    print(f"\nDataset shape: {df.shape}")
    print(f"Fraud rate: {df['isFraud'].mean():.4f}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Categorical columns: {len(df.select_dtypes(include=['object']).columns)}")
    print(f"Numerical columns: {len(df.select_dtypes(include=[np.number]).columns)}")
    print("\nEDA Complete! Check outputs/figures/ for visualizations.")

if __name__ == "__main__":
    main()

