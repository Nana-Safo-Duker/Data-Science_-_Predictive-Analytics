"""
Exploratory Data Analysis Script for Fuel Consumption Dataset
This script performs comprehensive EDA and saves visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_data():
    """Load the fuel consumption dataset"""
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'FuelConsumption.csv')
    df = pd.read_csv(data_path)
    # Clean column names (remove trailing spaces)
    df.columns = df.columns.str.strip()
    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    return df

def data_overview(df):
    """Display basic data overview"""
    print("\n" + "="*50)
    print("DATA OVERVIEW")
    print("="*50)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nDescriptive statistics:")
    print(df.describe(include='all'))

def data_quality_check(df):
    """Check data quality"""
    print("\n" + "="*50)
    print("DATA QUALITY ASSESSMENT")
    print("="*50)
    
    # Missing values
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_values,
        'Missing Percentage': missing_percent
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    if len(missing_df) > 0:
        print("\nMissing Values:")
        print(missing_df)
    else:
        print("\n✓ No missing values found!")
    
    # Duplicates
    duplicate_count = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicate_count}")
    
    return missing_df, duplicate_count

def clean_data(df):
    """Clean the dataset"""
    print("\n" + "="*50)
    print("DATA CLEANING")
    print("="*50)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    print("✓ Column names cleaned!")
    
    return df

def analyze_distributions(df, output_dir):
    """Analyze distributions of numerical variables"""
    print("\n" + "="*50)
    print("DISTRIBUTION ANALYSIS")
    print("="*50)
    
    numerical_cols = ['ENGINE SIZE', 'CYLINDERS', 'FUEL CONSUMPTION', 'COEMISSIONS']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Histograms
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(numerical_cols):
        axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Distribution plots saved!")
    
    # Box plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(numerical_cols):
        axes[idx].boxplot(df[col].dropna())
        axes[idx].set_title(f'Box Plot of {col}')
        axes[idx].set_ylabel(col)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Box plots saved!")

def analyze_categorical(df, output_dir):
    """Analyze categorical variables"""
    print("\n" + "="*50)
    print("CATEGORICAL VARIABLE ANALYSIS")
    print("="*50)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # MAKE
    top_makes = df['MAKE'].value_counts().head(15)
    axes[0, 0].barh(top_makes.index, top_makes.values)
    axes[0, 0].set_title('Top 15 Vehicle Makes')
    axes[0, 0].set_xlabel('Count')
    
    # VEHICLE CLASS
    vehicle_class = df['VEHICLE CLASS'].value_counts()
    axes[0, 1].barh(vehicle_class.index, vehicle_class.values)
    axes[0, 1].set_title('Vehicle Class Distribution')
    axes[0, 1].set_xlabel('Count')
    
    # FUEL
    fuel = df['FUEL'].value_counts()
    axes[1, 0].bar(fuel.index, fuel.values)
    axes[1, 0].set_title('Fuel Type Distribution')
    axes[1, 0].set_xlabel('Fuel Type')
    axes[1, 0].set_ylabel('Count')
    
    # CYLINDERS
    cylinders = df['CYLINDERS'].value_counts().sort_index()
    axes[1, 1].bar(cylinders.index.astype(str), cylinders.values)
    axes[1, 1].set_title('Cylinders Distribution')
    axes[1, 1].set_xlabel('Number of Cylinders')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'categorical_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Categorical analysis plots saved!")

def analyze_correlation(df, output_dir):
    """Analyze correlations"""
    print("\n" + "="*50)
    print("CORRELATION ANALYSIS")
    print("="*50)
    
    numerical_cols = ['ENGINE SIZE', 'CYLINDERS', 'FUEL CONSUMPTION', 'COEMISSIONS']
    correlation_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Numerical Variables')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Correlation matrix saved!")
    
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

def analyze_temporal_trends(df, output_dir):
    """Analyze trends over time"""
    print("\n" + "="*50)
    print("TEMPORAL TREND ANALYSIS")
    print("="*50)
    
    yearly_stats = df.groupby('Year').agg({
        'FUEL CONSUMPTION': ['mean', 'median', 'std'],
        'COEMISSIONS': ['mean', 'median', 'std'],
        'ENGINE SIZE': 'mean'
    }).round(2)
    print("\nYearly Statistics:")
    print(yearly_stats)
    
    # Plot trends
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    yearly_mean = df.groupby('Year')['FUEL CONSUMPTION'].mean()
    axes[0].plot(yearly_mean.index, yearly_mean.values, marker='o', linewidth=2, markersize=8)
    axes[0].set_title('Average Fuel Consumption Over Years')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Average Fuel Consumption')
    axes[0].grid(True, alpha=0.3)
    
    yearly_co2 = df.groupby('Year')['COEMISSIONS'].mean()
    axes[1].plot(yearly_co2.index, yearly_co2.values, marker='s', color='red', linewidth=2, markersize=8)
    axes[1].set_title('Average CO2 Emissions Over Years')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Average CO2 Emissions')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'yearly_trends.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Temporal trend plots saved!")

def detect_outliers(df):
    """Detect outliers using IQR method"""
    print("\n" + "="*50)
    print("OUTLIER DETECTION")
    print("="*50)
    
    numerical_cols = ['ENGINE SIZE', 'CYLINDERS', 'FUEL CONSUMPTION', 'COEMISSIONS']
    
    def detect_outliers_iqr(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        return outliers, lower_bound, upper_bound
    
    for col in numerical_cols:
        outliers, lower, upper = detect_outliers_iqr(df, col)
        print(f"\n{col}:")
        print(f"  Lower bound: {lower:.2f}, Upper bound: {upper:.2f}")
        print(f"  Number of outliers: {len(outliers)}")
        print(f"  Percentage: {(len(outliers)/len(df))*100:.2f}%")

def generate_summary(df):
    """Generate summary insights"""
    print("\n" + "="*50)
    print("SUMMARY INSIGHTS")
    print("="*50)
    print(f"\n1. Dataset contains {len(df)} records with {len(df.columns)} features")
    print(f"2. Time period: {df['Year'].min()} - {df['Year'].max()}")
    print(f"3. Number of unique makes: {df['MAKE'].nunique()}")
    print(f"4. Number of unique models: {df['MODEL'].nunique()}")
    print(f"5. Average fuel consumption: {df['FUEL CONSUMPTION'].mean():.2f}")
    print(f"6. Average CO2 emissions: {df['COEMISSIONS'].mean():.2f}")
    print(f"7. Strongest correlation: Fuel Consumption vs CO2 Emissions = {df['FUEL CONSUMPTION'].corr(df['COEMISSIONS']):.3f}")

def main():
    """Main function"""
    print("="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("Fuel Consumption Dataset")
    print("="*50)
    
    # Load data
    df = load_data()
    
    # Data overview
    data_overview(df)
    
    # Data quality check
    data_quality_check(df)
    
    # Clean data
    df = clean_data(df)
    
    # Set output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze distributions
    analyze_distributions(df, output_dir)
    
    # Analyze categorical variables
    analyze_categorical(df, output_dir)
    
    # Analyze correlations
    analyze_correlation(df, output_dir)
    
    # Analyze temporal trends
    analyze_temporal_trends(df, output_dir)
    
    # Detect outliers
    detect_outliers(df)
    
    # Generate summary
    generate_summary(df)
    
    # Save cleaned dataset
    output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'FuelConsumption_cleaned.csv')
    df.to_csv(output_path, index=False)
    print("\n✓ Cleaned dataset saved!")
    
    print("\n" + "="*50)
    print("EDA COMPLETE!")
    print("="*50)

if __name__ == "__main__":
    main()




