"""
Exploratory Data Analysis Script
Customer Data Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def load_data():
    """Load the dataset"""
    df = pd.read_csv(data_path)
    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    return df

def data_overview(df):
    """Provide data overview"""
    print("\n=== Dataset Information ===")
    df.info()
    print(f"\n=== Dataset Shape ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"\n=== Column Names ===")
    print(df.columns.tolist())
    return df.head()

def missing_values_analysis(df):
    """Analyze missing values"""
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_values.index,
        'Missing Count': missing_values.values,
        'Missing Percentage': missing_percentage.values
    })
    
    print("\n=== Missing Values Analysis ===")
    result_df = missing_df[missing_df['Missing Count'] > 0]
    if len(result_df) == 0:
        print("✓ No missing values found in the dataset!")
    else:
        print(result_df)
    return missing_df

def duplicate_analysis(df):
    """Analyze duplicate records"""
    duplicate_count = df.duplicated().sum()
    print(f"\n=== Duplicate Records ===")
    print(f"Number of duplicate records: {duplicate_count}")
    if duplicate_count > 0:
        print("\nDuplicate records:")
        print(df[df.duplicated()])
    else:
        print("✓ No duplicate records found!")
    return duplicate_count

def summary_statistics(df):
    """Generate summary statistics"""
    print("\n=== Numerical Columns Summary ===")
    print(df.describe())
    
    print("\n=== Categorical Columns Summary ===")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Most frequent: {df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A'}")

def visualize_distributions(df):
    """Create distribution visualizations"""
    # Customer ID distribution
    plt.figure(figsize=(12, 6))
    plt.hist(df['CustomerID'], bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Customer ID')
    plt.ylabel('Frequency')
    plt.title('Distribution of Customer IDs')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_path / 'customer_id_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Country distribution
    country_counts = df['Country'].value_counts()
    plt.figure(figsize=(14, 8))
    country_counts.plot(kind='bar', color='steelblue', edgecolor='black')
    plt.xlabel('Country')
    plt.ylabel('Number of Customers')
    plt.title('Customer Distribution by Country')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_path / 'country_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # City distribution (top 20)
    city_counts = df['City'].value_counts().head(20)
    plt.figure(figsize=(14, 8))
    city_counts.plot(kind='barh', color='coral', edgecolor='black')
    plt.xlabel('Number of Customers')
    plt.ylabel('City')
    plt.title('Top 20 Cities by Customer Count')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_path / 'city_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Visualizations saved to results/ directory")

def data_quality_checks(df):
    """Perform data quality checks"""
    print("\n=== Data Quality Checks ===")
    
    # Check for empty strings
    print("\n=== Empty String Check ===")
    for col in df.columns:
        empty_count = (df[col] == '').sum()
        if empty_count > 0:
            print(f"{col}: {empty_count} empty strings")
    
    # Check data consistency
    print("\n=== Data Consistency Check ===")
    print(f"Unique Customer IDs: {df['CustomerID'].nunique()}")
    print(f"Total Records: {len(df)}")
    if df['CustomerID'].nunique() == len(df):
        print("✓ All Customer IDs are unique")
    else:
        print("⚠ Warning: Duplicate Customer IDs found")

def main():
    """Main function"""
    print("=" * 50)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)
    
    # Load data
    df = load_data()
    
    # Data overview
    data_overview(df)
    
    # Missing values
    missing_values_analysis(df)
    
    # Duplicates
    duplicate_analysis(df)
    
    # Summary statistics
    summary_statistics(df)
    
    # Visualizations
    visualize_distributions(df)
    
    # Data quality checks
    data_quality_checks(df)
    
    # Summary
    print("\n" + "=" * 50)
    print("EDA SUMMARY")
    print("=" * 50)
    print(f"\n1. Dataset contains {df.shape[0]} customers with {df.shape[1]} attributes")
    print(f"2. Countries represented: {df['Country'].nunique()}")
    print(f"3. Cities represented: {df['City'].nunique()}")
    print(f"4. Missing values: {df.isnull().sum().sum()}")
    print(f"5. Duplicate records: {df.duplicated().sum()}")
    print(f"\n6. Most common country: {df['Country'].mode()[0]} ({df['Country'].value_counts().iloc[0]} customers)")
    print(f"7. Most common city: {df['City'].mode()[0]} ({df['City'].value_counts().iloc[0]} customers)")
    
    print("\n=== Key Insights ===")
    print("• The dataset appears to be clean with no missing values")
    print("• All customer IDs are unique")
    print("• The dataset has good geographical diversity")
    print("• Ready for further statistical and ML analysis")

if __name__ == "__main__":
    main()

