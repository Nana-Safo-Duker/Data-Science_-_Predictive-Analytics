"""
Comprehensive Exploratory Data Analysis (EDA) for Unicorn Companies Dataset

This script performs:
1. Data loading and understanding
2. Data cleaning and preprocessing
3. Missing value analysis
4. Data type analysis
5. Basic statistics and summary
6. Data visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Create results directory if it doesn't exist
os.makedirs('../../results/plots', exist_ok=True)
os.makedirs('../../data', exist_ok=True)


def load_data(filepath='../../data/Unicorn_Companies.csv'):
    """Load the dataset"""
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"Dataset Shape: {df.shape}")
    return df


def clean_data(df):
    """Clean and preprocess the data"""
    print("\nCleaning data...")
    
    # Fix column name typo
    df.rename(columns={'Select Inverstors': 'Select Investors'}, inplace=True)
    
    # Clean Valuation column
    def clean_valuation(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).replace('$', '').strip()
        try:
            return float(val_str)
        except:
            return np.nan
    
    df['Valuation_B'] = df['Valuation ($B)'].apply(clean_valuation)
    
    # Clean Total Raised column
    def clean_total_raised(val):
        if pd.isna(val) or val == 'None':
            return np.nan
        val_str = str(val)
        val_str = val_str.replace('$', '').strip()
        
        if 'B' in val_str.upper():
            return float(val_str.upper().replace('B', '').strip())
        elif 'M' in val_str.upper():
            return float(val_str.upper().replace('M', '').strip()) / 1000
        elif 'K' in val_str.upper():
            return float(val_str.upper().replace('K', '').strip()) / 1000000
        else:
            try:
                return float(val_str) / 1000000000
            except:
                return np.nan
    
    df['Total_Raised_B'] = df['Total Raised'].apply(clean_total_raised)
    
    # Convert Date Joined to datetime
    df['Date_Joined'] = pd.to_datetime(df['Date Joined'], errors='coerce')
    df['Year_Joined'] = df['Date_Joined'].dt.year
    
    # Clean Founded Year
    def clean_founded_year(val):
        if pd.isna(val) or val == 'None':
            return np.nan
        try:
            return int(float(val))
        except:
            return np.nan
    
    df['Founded_Year'] = df['Founded Year'].apply(clean_founded_year)
    
    # Calculate years to unicorn status
    df['Years_to_Unicorn'] = df['Year_Joined'] - df['Founded_Year']
    
    print("Data cleaning completed!")
    return df


def analyze_missing_values(df):
    """Analyze missing values in the dataset"""
    print("\n" + "="*50)
    print("Missing Value Analysis")
    print("="*50)
    
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_values.index,
        'Missing Count': missing_values.values,
        'Missing Percentage': missing_percent.values
    })
    
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df.to_string(index=False))
    else:
        print("No missing values found!")
    
    return missing_df


def basic_statistics(df):
    """Generate basic statistical summary"""
    print("\n" + "="*50)
    print("Basic Statistical Summary")
    print("="*50)
    
    numeric_cols = ['Valuation_B', 'Total_Raised_B', 'Investors Count', 'Deal Terms', 
                    'Portfolio Exits', 'Founded_Year', 'Year_Joined', 'Years_to_Unicorn']
    
    print("\nNumerical Columns Summary:")
    print(df[numeric_cols].describe())
    
    print("\n" + "="*50)
    print("Categorical Columns Summary")
    print("="*50)
    
    categorical_cols = ['Country', 'City', 'Industry', 'Financial Stage']
    
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts().head(10))


def create_visualizations(df):
    """Create comprehensive visualizations"""
    print("\n" + "="*50)
    print("Creating Visualizations")
    print("="*50)
    
    # Distribution of Valuations
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    df['Valuation_B'].hist(bins=50, edgecolor='black')
    plt.title('Distribution of Company Valuations')
    plt.xlabel('Valuation ($B)')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    df['Valuation_B'].plot(kind='box')
    plt.title('Box Plot of Company Valuations')
    plt.ylabel('Valuation ($B)')
    
    plt.tight_layout()
    plt.savefig('../../results/plots/valuation_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: valuation_distribution.png")
    
    # Top 10 Countries
    plt.figure(figsize=(12, 6))
    top_countries = df['Country'].value_counts().head(10)
    sns.barplot(x=top_countries.values, y=top_countries.index)
    plt.title('Top 10 Countries by Number of Unicorn Companies')
    plt.xlabel('Number of Unicorns')
    plt.ylabel('Country')
    plt.tight_layout()
    plt.savefig('../../results/plots/top_countries.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: top_countries.png")
    
    # Top Industries
    plt.figure(figsize=(12, 8))
    top_industries = df['Industry'].value_counts().head(15)
    sns.barplot(x=top_industries.values, y=top_industries.index)
    plt.title('Top 15 Industries by Number of Unicorn Companies')
    plt.xlabel('Number of Unicorns')
    plt.ylabel('Industry')
    plt.tight_layout()
    plt.savefig('../../results/plots/top_industries.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: top_industries.png")
    
    # Unicorns by Year
    plt.figure(figsize=(14, 6))
    yearly_unicorns = df.groupby('Year_Joined').size()
    yearly_unicorns.plot(kind='line', marker='o')
    plt.title('Number of Unicorn Companies by Year Joined')
    plt.xlabel('Year')
    plt.ylabel('Number of Unicorns')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../../results/plots/unicorns_by_year.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: unicorns_by_year.png")


def save_cleaned_data(df, filepath='../../data/Unicorn_Companies_cleaned.csv'):
    """Save cleaned dataset"""
    df.to_csv(filepath, index=False)
    print(f"\nCleaned dataset saved to {filepath}")


def main():
    """Main function to run EDA"""
    print("="*50)
    print("Exploratory Data Analysis: Unicorn Companies")
    print("="*50)
    
    # Load data
    df = load_data()
    
    # Display basic info
    print("\nDataset Info:")
    df.info()
    
    print("\nColumn Names:")
    print(df.columns.tolist())
    
    # Analyze missing values
    analyze_missing_values(df)
    
    # Clean data
    df_cleaned = clean_data(df.copy())
    
    # Basic statistics
    basic_statistics(df_cleaned)
    
    # Create visualizations
    create_visualizations(df_cleaned)
    
    # Save cleaned data
    save_cleaned_data(df_cleaned)
    
    print("\n" + "="*50)
    print("EDA Completed Successfully!")
    print("="*50)


if __name__ == "__main__":
    main()
