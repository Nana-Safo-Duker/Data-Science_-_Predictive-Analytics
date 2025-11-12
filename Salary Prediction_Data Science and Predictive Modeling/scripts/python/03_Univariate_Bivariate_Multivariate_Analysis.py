"""
Univariate, Bivariate, and Multivariate Analysis Script
This script performs comprehensive univariate, bivariate, and multivariate analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import linregress, pearsonr, spearmanr
import warnings
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
warnings.filterwarnings('ignore')

def load_data():
    """Load the dataset."""
    data_path = project_root / "data" / "raw" / "Position_Salaries.csv"
    df = pd.read_csv(data_path)
    return df

def univariate_analysis(df, output_dir):
    """Perform univariate analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("UNIVARIATE ANALYSIS")
    print("=" * 60)
    
    # Salary analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    axes[0, 0].hist(df['Salary'], bins=10, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Histogram of Salary', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Salary', fontsize=10)
    axes[0, 0].set_ylabel('Frequency', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].boxplot(df['Salary'], vert=True)
    axes[0, 1].set_title('Box Plot of Salary', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Salary', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    df['Salary'].plot.density(ax=axes[0, 2], color='green', linewidth=2)
    axes[0, 2].set_title('Density Plot of Salary', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Salary', fontsize=10)
    axes[0, 2].set_ylabel('Density', fontsize=10)
    axes[0, 2].grid(True, alpha=0.3)
    
    stats.probplot(df['Salary'], dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot of Salary', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].bar(df['Level'], df['Salary'], color='steelblue', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Salary by Level', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Level', fontsize=10)
    axes[1, 1].set_ylabel('Salary', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    axes[1, 2].axis('off')
    stats_text = f"""
    Salary Statistics:
    Mean: ${df['Salary'].mean():,.2f}
    Median: ${df['Salary'].median():,.2f}
    Std Dev: ${df['Salary'].std():,.2f}
    Min: ${df['Salary'].min():,.2f}
    Max: ${df['Salary'].max():,.2f}
    """
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'univariate_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def bivariate_analysis(df, output_dir):
    """Perform bivariate analysis."""
    output_dir = Path(output_dir)
    
    print("=" * 60)
    print("BIVARIATE ANALYSIS")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    axes[0, 0].scatter(df['Level'], df['Salary'], s=150, alpha=0.7, color='coral', edgecolors='black')
    axes[0, 0].set_title('Scatter Plot: Level vs Salary', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Level', fontsize=10)
    axes[0, 0].set_ylabel('Salary', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    slope, intercept, r_value, p_value, std_err = linregress(df['Level'], df['Salary'])
    line = slope * df['Level'] + intercept
    axes[0, 1].scatter(df['Level'], df['Salary'], s=150, alpha=0.7, color='purple', edgecolors='black')
    axes[0, 1].plot(df['Level'], line, 'r--', linewidth=2, label=f'Linear Fit (r²={r_value**2:.4f})')
    axes[0, 1].set_title('Scatter Plot with Regression Line', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Level', fontsize=10)
    axes[0, 1].set_ylabel('Salary', fontsize=10)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    residuals = df['Salary'] - (slope * df['Level'] + intercept)
    axes[0, 2].scatter(df['Level'], residuals, s=100, alpha=0.7, color='orange', edgecolors='black')
    axes[0, 2].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 2].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Level', fontsize=10)
    axes[0, 2].set_ylabel('Residuals', fontsize=10)
    axes[0, 2].grid(True, alpha=0.3)
    
    pearson_corr, pearson_p = pearsonr(df['Level'], df['Salary'])
    spearman_corr, spearman_p = spearmanr(df['Level'], df['Salary'])
    
    axes[1, 0].axis('off')
    stats_text = f"""
    Bivariate Statistics:
    Pearson Correlation: {pearson_corr:.4f}
    Pearson p-value: {pearson_p:.6f}
    Spearman Correlation: {spearman_corr:.4f}
    Spearman p-value: {spearman_p:.6f}
    R²: {r_value**2:.4f}
    """
    axes[1, 0].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bivariate_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def multivariate_analysis(df, output_dir):
    """Perform multivariate analysis."""
    output_dir = Path(output_dir)
    
    print("=" * 60)
    print("MULTIVARIATE ANALYSIS")
    print("=" * 60)
    
    # Create additional features
    df_analysis = df.copy()
    df_analysis['Salary_Category'] = pd.cut(df_analysis['Salary'], 
                                             bins=[0, 80000, 150000, 300000, float('inf')],
                                             labels=['Low', 'Medium', 'High', 'Very High'])
    df_analysis['Level_Group'] = pd.cut(df_analysis['Level'], 
                                         bins=[0, 3, 6, 10],
                                         labels=['Junior', 'Mid', 'Senior'])
    df_analysis['Log_Salary'] = np.log(df_analysis['Salary'])
    
    # Correlation heatmap
    numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns
    correlation_matrix = df_analysis[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=2, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap (Multivariate)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'multivariate_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save analysis data
    analysis_path = project_root / 'data' / 'processed' / 'analysis_data.csv'
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    df_analysis.to_csv(analysis_path, index=False)
    
    return df_analysis

def main():
    """Main function."""
    df = load_data()
    output_dir = project_root / 'results' / 'figures'
    
    univariate_analysis(df, output_dir)
    bivariate_analysis(df, output_dir)
    df_analysis = multivariate_analysis(df, output_dir)
    
    print("\nAnalysis completed successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"Analysis data saved to: {project_root / 'data' / 'processed' / 'analysis_data.csv'}")

if __name__ == "__main__":
    main()


