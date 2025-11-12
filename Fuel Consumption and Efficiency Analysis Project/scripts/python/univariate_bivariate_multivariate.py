"""
Univariate, Bivariate, and Multivariate Analysis Script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

def load_data():
    """Load the dataset"""
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'FuelConsumption.csv')
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    return df

def univariate_analysis(df, output_dir):
    """Perform univariate analysis"""
    print("="*50)
    print("UNIVARIATE ANALYSIS")
    print("="*50)
    
    numerical_cols = ['ENGINE SIZE', 'CYLINDERS', 'FUEL CONSUMPTION', 'COEMISSIONS']
    
    # Statistics
    print("\nUnivariate Statistics:\n")
    for col in numerical_cols:
        print(f"{col}:")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Median: {df[col].median():.2f}")
        print(f"  Std: {df[col].std():.2f}")
        print(f"  Skewness: {df[col].skew():.2f}")
        print(f"  Kurtosis: {df[col].kurtosis():.2f}\n")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for idx, col in enumerate(numerical_cols):
        axes[idx].hist(df[col], bins=30, alpha=0.7, edgecolor='black', density=True)
        kde = gaussian_kde(df[col].dropna())
        x_range = np.linspace(df[col].min(), df[col].max(), 100)
        axes[idx].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Density')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'univariate_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Univariate analysis plots saved!")

def bivariate_analysis(df, output_dir):
    """Perform bivariate analysis"""
    print("\n" + "="*50)
    print("BIVARIATE ANALYSIS")
    print("="*50)
    
    # Correlation coefficients
    print("\nBivariate Correlation Analysis:\n")
    target = 'COEMISSIONS'
    for col in ['ENGINE SIZE', 'CYLINDERS', 'FUEL CONSUMPTION']:
        corr = df[col].corr(df[target])
        print(f"{col} vs {target}: r = {corr:.4f}")
    
    # Scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    # Engine Size vs Fuel Consumption
    axes[0].scatter(df['ENGINE SIZE'], df['FUEL CONSUMPTION'], alpha=0.5)
    z = np.polyfit(df['ENGINE SIZE'], df['FUEL CONSUMPTION'], 1)
    p = np.poly1d(z)
    axes[0].plot(df['ENGINE SIZE'], p(df['ENGINE SIZE']), "r--", alpha=0.8)
    axes[0].set_xlabel('Engine Size')
    axes[0].set_ylabel('Fuel Consumption')
    axes[0].set_title('Engine Size vs Fuel Consumption')
    axes[0].grid(True, alpha=0.3)
    
    # Cylinders vs Fuel Consumption
    axes[1].scatter(df['CYLINDERS'], df['FUEL CONSUMPTION'], alpha=0.5)
    axes[1].set_xlabel('Cylinders')
    axes[1].set_ylabel('Fuel Consumption')
    axes[1].set_title('Cylinders vs Fuel Consumption')
    axes[1].grid(True, alpha=0.3)
    
    # Fuel Consumption vs CO2 Emissions
    axes[2].scatter(df['FUEL CONSUMPTION'], df['COEMISSIONS'], alpha=0.5, color='green')
    z = np.polyfit(df['FUEL CONSUMPTION'], df['COEMISSIONS'], 1)
    p = np.poly1d(z)
    axes[2].plot(df['FUEL CONSUMPTION'], p(df['FUEL CONSUMPTION']), "r--", alpha=0.8)
    axes[2].set_xlabel('Fuel Consumption')
    axes[2].set_ylabel('CO2 Emissions')
    axes[2].set_title('Fuel Consumption vs CO2 Emissions')
    axes[2].grid(True, alpha=0.3)
    
    # Engine Size vs CO2 Emissions
    axes[3].scatter(df['ENGINE SIZE'], df['COEMISSIONS'], alpha=0.5, color='orange')
    z = np.polyfit(df['ENGINE SIZE'], df['COEMISSIONS'], 1)
    p = np.poly1d(z)
    axes[3].plot(df['ENGINE SIZE'], p(df['ENGINE SIZE']), "r--", alpha=0.8)
    axes[3].set_xlabel('Engine Size')
    axes[3].set_ylabel('CO2 Emissions')
    axes[3].set_title('Engine Size vs CO2 Emissions')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bivariate_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Bivariate analysis plots saved!")

def multivariate_analysis(df, output_dir):
    """Perform multivariate analysis"""
    print("\n" + "="*50)
    print("MULTIVARIATE ANALYSIS")
    print("="*50)
    
    numerical_cols = ['ENGINE SIZE', 'CYLINDERS', 'FUEL CONSUMPTION', 'COEMISSIONS']
    
    # Pair plot
    sns.pairplot(df[numerical_cols], diag_kind='kde', plot_kws={'alpha': 0.6})
    plt.suptitle('Pair Plot - Multivariate Analysis', y=1.02)
    plt.savefig(os.path.join(output_dir, 'multivariate_pairplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Multivariate pair plot saved!")
    
    # Correlation heatmap
    correlation_matrix = df[numerical_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.3f')
    plt.title('Multivariate Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multivariate_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Multivariate correlation matrix saved!")

def main():
    """Main function"""
    df = load_data()
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    univariate_analysis(df, output_dir)
    bivariate_analysis(df, output_dir)
    multivariate_analysis(df, output_dir)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE!")
    print("="*50)

if __name__ == "__main__":
    main()


