"""
Exploratory Data Analysis for Cybersecurity Attacks Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import missingno as msno
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

def load_data(file_path):
    """Load the cybersecurity attacks dataset"""
    df = pd.read_csv(file_path)
    print(f"Dataset Shape: {df.shape}")
    return df

def analyze_missing_values(df):
    """Analyze missing values in the dataset"""
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_values.index,
        'Missing Count': missing_values.values,
        'Missing Percentage': missing_percentage.values
    })
    
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    return missing_df

def clean_data(df):
    """Clean and preprocess the dataset"""
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Remove the '.' column if it exists
    if '.' in df.columns:
        df = df.drop(columns=['.'])
    
    # Clean Attack category
    if 'Attack category' in df.columns:
        df['Attack category'] = df['Attack category'].str.strip()
    
    return df

def parse_time_column(df):
    """Parse the Time column"""
    if 'Time' in df.columns:
        def parse_time(time_str):
            if pd.isna(time_str):
                return None, None
            try:
                if '-' in str(time_str):
                    start, end = str(time_str).split('-')
                    return int(start), int(end)
                else:
                    return int(time_str), int(time_str)
            except:
                return None, None
        
        time_parsed = df['Time'].apply(parse_time)
        df['Time_Start'] = [t[0] for t in time_parsed]
        df['Time_End'] = [t[1] for t in time_parsed]
        df['Time_Duration'] = df['Time_End'] - df['Time_Start']
        
        # Convert to datetime
        df['Datetime_Start'] = pd.to_datetime(df['Time_Start'], unit='s', errors='coerce')
        df['Datetime_End'] = pd.to_datetime(df['Time_End'], unit='s', errors='coerce')
        
        # Extract temporal features
        df['Date'] = df['Datetime_Start'].dt.date
        df['Hour'] = df['Datetime_Start'].dt.hour
        df['DayOfWeek'] = df['Datetime_Start'].dt.day_name()
        df['Month'] = df['Datetime_Start'].dt.month
    
    return df

def main():
    """Main function to run EDA"""
    # Load data
    file_path = '../../data/Cybersecurity_attacks.csv'
    df = load_data(file_path)
    
    # Analyze missing values
    missing_df = analyze_missing_values(df)
    print("\nMissing Values Summary:")
    print(missing_df)
    
    # Clean data
    df = clean_data(df)
    df = parse_time_column(df)
    
    # Generate summary statistics
    print("\n=== DATASET SUMMARY ===")
    print(f"Total Attacks: {len(df):,}")
    print(f"Unique Attack Categories: {df['Attack category'].nunique() if 'Attack category' in df.columns else 0}")
    print(f"Unique Protocols: {df['Protocol'].nunique() if 'Protocol' in df.columns else 0}")
    
    # Save cleaned dataset
    output_path = '../../data/cybersecurity_attacks_cleaned.csv'
    df.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved to {output_path}")

if __name__ == "__main__":
    main()



