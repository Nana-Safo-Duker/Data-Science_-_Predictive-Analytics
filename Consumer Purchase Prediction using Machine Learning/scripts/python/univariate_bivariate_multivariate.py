"""
Univariate, Bivariate, and Multivariate Analysis Script
Consumer Purchase Prediction Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load the dataset"""
    df = pd.read_csv(file_path)
    return df

def univariate_analysis(df, output_dir='../../output'):
    """Perform univariate analysis"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    numeric_cols = ['Age', 'EstimatedSalary']
    
    for col in numeric_cols:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histogram
        axes[0, 0].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title(f'{col} Distribution (Histogram)')
        axes[0, 0].set_xlabel(col)
        axes[0, 0].set_ylabel('Frequency')
        
        # Box plot
        axes[0, 1].boxplot(df[col], vert=True)
        axes[0, 1].set_title(f'{col} Distribution (Box Plot)')
        axes[0, 1].set_ylabel(col)
        
        # Q-Q plot
        stats.probplot(df[col], dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title(f'{col} Q-Q Plot')
        
        # Density plot
        df[col].plot.density(ax=axes[1, 1])
        axes[1, 1].set_title(f'{col} Density Plot')
        axes[1, 1].set_xlabel(col)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/univariate_{col.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n{col} Statistics:")
        print(f"Mean: {df[col].mean():.2f}")
        print(f"Median: {df[col].median():.2f}")
        print(f"Std: {df[col].std():.2f}")
        print(f"Skewness: {df[col].skew():.3f}")
        print(f"Kurtosis: {df[col].kurtosis():.3f}")

def bivariate_analysis(df, output_dir='../../output'):
    """Perform bivariate analysis"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Age vs Purchased
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.boxplot(data=df, x='Purchased', y='Age', ax=axes[0])
    axes[0].set_title('Age by Purchase Status')
    axes[0].set_xticklabels(['No', 'Yes'])
    
    sns.violinplot(data=df, x='Purchased', y='Age', ax=axes[1])
    axes[1].set_title('Age Distribution by Purchase Status (Violin)')
    axes[1].set_xticklabels(['No', 'Yes'])
    
    sns.stripplot(data=df, x='Purchased', y='Age', ax=axes[2], alpha=0.5)
    axes[2].set_title('Age Distribution by Purchase Status (Strip)')
    axes[2].set_xticklabels(['No', 'Yes'])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bivariate_age_purchased.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Salary vs Purchased
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.boxplot(data=df, x='Purchased', y='EstimatedSalary', ax=axes[0])
    axes[0].set_title('Estimated Salary by Purchase Status')
    axes[0].set_xticklabels(['No', 'Yes'])
    
    sns.violinplot(data=df, x='Purchased', y='EstimatedSalary', ax=axes[1])
    axes[1].set_title('Estimated Salary Distribution by Purchase Status (Violin)')
    axes[1].set_xticklabels(['No', 'Yes'])
    
    sns.stripplot(data=df, x='Purchased', y='EstimatedSalary', ax=axes[2], alpha=0.5)
    axes[2].set_title('Estimated Salary Distribution by Purchase Status (Strip)')
    axes[2].set_xticklabels(['No', 'Yes'])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bivariate_salary_purchased.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Age vs Salary
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['Age'], df['EstimatedSalary'], 
                         c=df['Purchased'], cmap='coolwarm', 
                         alpha=0.6, s=50, edgecolors='black')
    plt.colorbar(scatter, label='Purchased')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.title('Age vs Estimated Salary (colored by Purchase Status)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bivariate_age_salary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Statistical tests
    age_0 = df[df['Purchased'] == 0]['Age']
    age_1 = df[df['Purchased'] == 1]['Age']
    t_stat, p_value = ttest_ind(age_0, age_1)
    print(f"\nAge T-test: t-statistic={t_stat:.4f}, p-value={p_value:.4f}")
    
    salary_0 = df[df['Purchased'] == 0]['EstimatedSalary']
    salary_1 = df[df['Purchased'] == 1]['EstimatedSalary']
    t_stat, p_value = ttest_ind(salary_0, salary_1)
    print(f"Salary T-test: t-statistic={t_stat:.4f}, p-value={p_value:.4f}")

def multivariate_analysis(df, output_dir='../../output'):
    """Perform multivariate analysis"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Pairwise relationships
    sns.pairplot(df[['Age', 'EstimatedSalary', 'Purchased']], 
                 hue='Purchased', diag_kind='kde')
    plt.suptitle('Pairwise Relationships', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/multivariate_pairplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Correlation matrix
    correlation_matrix = df[['Age', 'EstimatedSalary', 'Purchased']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/multivariate_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function"""
    df = load_data('../../data/Advertisement.csv')
    
    print("UNIVARIATE ANALYSIS")
    print("="*50)
    univariate_analysis(df)
    
    print("\nBIVARIATE ANALYSIS")
    print("="*50)
    bivariate_analysis(df)
    
    print("\nMULTIVARIATE ANALYSIS")
    print("="*50)
    multivariate_analysis(df)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    main()

