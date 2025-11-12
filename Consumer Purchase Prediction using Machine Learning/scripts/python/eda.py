"""
Exploratory Data Analysis Script
Consumer Purchase Prediction Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data(file_path):
    """Load the dataset"""
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully! Shape: {df.shape}")
    return df

def data_overview(df):
    """Perform initial data overview"""
    print("\n" + "="*50)
    print("DATA OVERVIEW")
    print("="*50)
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nDataset Info:")
    print(df.info())
    print(f"\nStatistical Summary:")
    print(df.describe())
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")

def analyze_target_variable(df):
    """Analyze target variable distribution"""
    print("\n" + "="*50)
    print("TARGET VARIABLE ANALYSIS")
    print("="*50)
    purchased_counts = df['Purchased'].value_counts()
    print(f"\nPurchased Distribution:")
    print(purchased_counts)
    print(f"\nPercentage:")
    print((purchased_counts / len(df)) * 100)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(purchased_counts.index, purchased_counts.values, color=['skyblue', 'coral'])
    axes[0].set_xlabel('Purchased')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Purchased Distribution')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['No', 'Yes'])
    
    axes[1].pie(purchased_counts.values, labels=['No', 'Yes'], autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Purchased Distribution (Pie Chart)')
    plt.tight_layout()
    plt.savefig('../../output/target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_numerical_variables(df):
    """Analyze numerical variables"""
    print("\n" + "="*50)
    print("NUMERICAL VARIABLES ANALYSIS")
    print("="*50)
    numeric_cols = ['Age', 'EstimatedSalary']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for idx, col in enumerate(numeric_cols):
        axes[idx, 0].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        axes[idx, 0].set_title(f'{col} Distribution (Histogram)')
        axes[idx, 0].set_xlabel(col)
        axes[idx, 0].set_ylabel('Frequency')
        
        axes[idx, 1].boxplot(df[col], vert=True)
        axes[idx, 1].set_title(f'{col} Distribution (Box Plot)')
        axes[idx, 1].set_ylabel(col)
    
    plt.tight_layout()
    plt.savefig('../../output/numerical_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_categorical_variables(df):
    """Analyze categorical variables"""
    print("\n" + "="*50)
    print("CATEGORICAL VARIABLES ANALYSIS")
    print("="*50)
    gender_counts = df['Gender'].value_counts()
    print(f"\nGender Distribution:")
    print(gender_counts)
    print(f"\nPercentage:")
    print((gender_counts / len(df)) * 100)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(gender_counts.index, gender_counts.values, color=['lightblue', 'lightpink'])
    axes[0].set_xlabel('Gender')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Gender Distribution')
    
    axes[1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Gender Distribution (Pie Chart)')
    plt.tight_layout()
    plt.savefig('../../output/categorical_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_relationships(df):
    """Analyze relationships between variables"""
    print("\n" + "="*50)
    print("RELATIONSHIP ANALYSIS")
    print("="*50)
    
    # Age vs Purchased
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.boxplot(data=df, x='Purchased', y='Age', ax=axes[0])
    axes[0].set_title('Age Distribution by Purchase Status')
    axes[0].set_xticklabels(['No', 'Yes'])
    
    sns.boxplot(data=df, x='Purchased', y='EstimatedSalary', ax=axes[1])
    axes[1].set_title('Salary Distribution by Purchase Status')
    axes[1].set_xticklabels(['No', 'Yes'])
    plt.tight_layout()
    plt.savefig('../../output/relationships.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Scatter plot
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
    plt.savefig('../../output/scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def correlation_analysis(df):
    """Perform correlation analysis"""
    print("\n" + "="*50)
    print("CORRELATION ANALYSIS")
    print("="*50)
    numeric_df = df[['Age', 'EstimatedSalary', 'Purchased']]
    correlation_matrix = numeric_df.corr()
    print(f"\nCorrelation Matrix:")
    print(correlation_matrix)
    print(f"\nCorrelation with Purchased:")
    print(correlation_matrix['Purchased'].sort_values(ascending=False))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('../../output/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run EDA"""
    # Create output directory
    import os
    os.makedirs('../../output', exist_ok=True)
    
    # Load data
    df = load_data('../../data/Advertisement.csv')
    
    # Perform analysis
    data_overview(df)
    analyze_target_variable(df)
    analyze_numerical_variables(df)
    analyze_categorical_variables(df)
    analyze_relationships(df)
    correlation_analysis(df)
    
    print("\n" + "="*50)
    print("EDA COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    main()

