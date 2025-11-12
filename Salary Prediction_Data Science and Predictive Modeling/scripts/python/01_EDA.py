"""
Exploratory Data Analysis (EDA) Script - Position Salaries Dataset
This script performs comprehensive EDA on the Position Salaries dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Set style and warnings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
warnings.filterwarnings('ignore')

def load_data():
    """Load the dataset from the raw data directory."""
    data_path = project_root / "data" / "raw" / "Position_Salaries.csv"
    df = pd.read_csv(data_path)
    return df

def data_overview(df):
    """Display basic overview of the dataset."""
    print("=" * 60)
    print("DATA OVERVIEW")
    print("=" * 60)
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    print(f"\nDataset Info:")
    print(df.info())
    print(f"\nDataset Description:")
    print(df.describe())
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")

def create_visualizations(df, output_dir):
    """Create and save all visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Salary Distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(df['Salary'], bins=10, edgecolor='black', alpha=0.7, color='skyblue')
    plt.title('Distribution of Salary', fontsize=14, fontweight='bold')
    plt.xlabel('Salary', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(df['Salary'], vert=True)
    plt.title('Box Plot of Salary', fontsize=14, fontweight='bold')
    plt.ylabel('Salary', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'salary_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Level vs Salary
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(df['Level'], df['Salary'], s=100, alpha=0.7, color='coral', edgecolors='black')
    plt.title('Level vs Salary', fontsize=14, fontweight='bold')
    plt.xlabel('Level', fontsize=12)
    plt.ylabel('Salary', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(df['Level'], df['Salary'], marker='o', linewidth=2, markersize=8, color='green')
    plt.title('Level vs Salary (Line Plot)', fontsize=14, fontweight='bold')
    plt.xlabel('Level', fontsize=12)
    plt.ylabel('Salary', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'level_vs_salary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Salary by Position
    plt.figure(figsize=(14, 8))
    df_sorted = df.sort_values('Salary')
    plt.barh(df_sorted['Position'], df_sorted['Salary'], color='steelblue', alpha=0.8)
    plt.title('Salary by Position', fontsize=16, fontweight='bold')
    plt.xlabel('Salary ($)', fontsize=12)
    plt.ylabel('Position', fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    
    for i, v in enumerate(df_sorted['Salary']):
        plt.text(v + 10000, i, f'${v:,}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'salary_by_position.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Correlation Heatmap
    correlation = df[['Level', 'Salary']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=2, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap: Level vs Salary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run EDA."""
    # Load data
    df = load_data()
    
    # Data overview
    data_overview(df)
    
    # Create visualizations
    output_dir = project_root / 'results' / 'figures'
    create_visualizations(df, output_dir)
    
    # Save processed data
    processed_path = project_root / 'data' / 'processed' / 'processed_data.csv'
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    
    print(f"\nEDA completed successfully!")
    print(f"Visualizations saved to: {output_dir}")
    print(f"Processed data saved to: {processed_path}")

if __name__ == "__main__":
    main()


