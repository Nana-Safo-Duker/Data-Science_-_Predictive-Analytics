"""
Univariate, Bivariate, and Multivariate Analysis for Unicorn Companies Dataset

This script performs:
1. Univariate Analysis - Single variable analysis
2. Bivariate Analysis - Two variable relationships
3. Multivariate Analysis - Multiple variable relationships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import warnings
import os
import sys

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Create results directory if it doesn't exist
os.makedirs('../../results/plots', exist_ok=True)


def load_cleaned_data(filepath='../../data/Unicorn_Companies_cleaned.csv'):
    """Load cleaned dataset"""
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("Cleaned dataset not found. Please run EDA script first.")
        sys.exit(1)


def univariate_analysis(df):
    """Perform univariate analysis"""
    print("\n" + "="*50)
    print("Univariate Analysis")
    print("="*50)
    
    # Univariate analysis for Valuation
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Histogram
    axes[0, 0].hist(df['Valuation_B'].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Distribution of Valuations')
    axes[0, 0].set_xlabel('Valuation ($B)')
    axes[0, 0].set_ylabel('Frequency')
    
    # Box plot
    axes[0, 1].boxplot(df['Valuation_B'].dropna())
    axes[0, 1].set_title('Box Plot of Valuations')
    axes[0, 1].set_ylabel('Valuation ($B)')
    
    # Q-Q plot
    stats.probplot(df['Valuation_B'].dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot - Normality Test')
    
    # Log transformation
    axes[1, 1].hist(np.log1p(df['Valuation_B'].dropna()), bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Log-Transformed Distribution of Valuations')
    axes[1, 1].set_xlabel('Log(Valuation)')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('../../results/plots/univariate_valuation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: univariate_valuation.png")
    
    # Univariate analysis for categorical variables
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Country distribution
    top_countries = df['Country'].value_counts().head(10)
    axes[0, 0].barh(top_countries.index, top_countries.values)
    axes[0, 0].set_title('Top 10 Countries')
    axes[0, 0].set_xlabel('Count')
    
    # Industry distribution
    top_industries = df['Industry'].value_counts().head(10)
    axes[0, 1].barh(top_industries.index, top_industries.values)
    axes[0, 1].set_title('Top 10 Industries')
    axes[0, 1].set_xlabel('Count')
    
    # Financial Stage distribution
    financial_stage = df['Financial Stage'].value_counts()
    axes[1, 0].pie(financial_stage.values, labels=financial_stage.index, autopct='%1.1f%%')
    axes[1, 0].set_title('Financial Stage Distribution')
    
    # Years to Unicorn distribution
    axes[1, 1].hist(df['Years_to_Unicorn'].dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Years to Unicorn Status')
    axes[1, 1].set_xlabel('Years')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('../../results/plots/univariate_categorical.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: univariate_categorical.png")


def bivariate_analysis(df):
    """Perform bivariate analysis"""
    print("\n" + "="*50)
    print("Bivariate Analysis")
    print("="*50)
    
    # Bivariate: Valuation vs Total Raised
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot
    axes[0].scatter(df['Total_Raised_B'], df['Valuation_B'], alpha=0.5)
    axes[0].set_xlabel('Total Raised ($B)')
    axes[0].set_ylabel('Valuation ($B)')
    axes[0].set_title('Valuation vs Total Raised')
    axes[0].grid(True, alpha=0.3)
    
    # Add regression line
    valid_data = df[['Total_Raised_B', 'Valuation_B']].dropna()
    if len(valid_data) > 0:
        z = np.polyfit(valid_data['Total_Raised_B'], valid_data['Valuation_B'], 1)
        p = np.poly1d(z)
        axes[0].plot(valid_data['Total_Raised_B'].sort_values(), 
                    p(valid_data['Total_Raised_B'].sort_values()), "r--", alpha=0.8)
    
    # Valuation vs Investors Count
    axes[1].scatter(df['Investors Count'], df['Valuation_B'], alpha=0.5)
    axes[1].set_xlabel('Investors Count')
    axes[1].set_ylabel('Valuation ($B)')
    axes[1].set_title('Valuation vs Investors Count')
    axes[1].grid(True, alpha=0.3)
    
    valid_data2 = df[['Investors Count', 'Valuation_B']].dropna()
    if len(valid_data2) > 0:
        z2 = np.polyfit(valid_data2['Investors Count'], valid_data2['Valuation_B'], 1)
        p2 = np.poly1d(z2)
        axes[1].plot(valid_data2['Investors Count'].sort_values(), 
                    p2(valid_data2['Investors Count'].sort_values()), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('../../results/plots/bivariate_valuation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: bivariate_valuation.png")
    
    # Bivariate: Valuation by Country
    top_5_countries = df['Country'].value_counts().head(5).index
    country_valuation = df[df['Country'].isin(top_5_countries)]
    
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=country_valuation, x='Country', y='Valuation_B')
    plt.title('Valuation Distribution by Country')
    plt.xlabel('Country')
    plt.ylabel('Valuation ($B)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../../results/plots/bivariate_country_valuation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: bivariate_country_valuation.png")
    
    # Bivariate: Valuation by Industry
    top_5_industries = df['Industry'].value_counts().head(5).index
    industry_valuation = df[df['Industry'].isin(top_5_industries)]
    
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=industry_valuation, x='Industry', y='Valuation_B')
    plt.title('Valuation Distribution by Industry')
    plt.xlabel('Industry')
    plt.ylabel('Valuation ($B)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../../results/plots/bivariate_industry_valuation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: bivariate_industry_valuation.png")
    
    # Bivariate: Years to Unicorn vs Valuation
    plt.figure(figsize=(12, 6))
    valid_data3 = df[['Years_to_Unicorn', 'Valuation_B']].dropna()
    plt.scatter(valid_data3['Years_to_Unicorn'], valid_data3['Valuation_B'], alpha=0.5)
    plt.xlabel('Years to Unicorn Status')
    plt.ylabel('Valuation ($B)')
    plt.title('Valuation vs Years to Unicorn Status')
    plt.grid(True, alpha=0.3)
    
    if len(valid_data3) > 0:
        z = np.polyfit(valid_data3['Years_to_Unicorn'], valid_data3['Valuation_B'], 1)
        p = np.poly1d(z)
        plt.plot(valid_data3['Years_to_Unicorn'].sort_values(), 
                p(valid_data3['Years_to_Unicorn'].sort_values()), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('../../results/plots/bivariate_years_valuation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: bivariate_years_valuation.png")


def multivariate_analysis(df):
    """Perform multivariate analysis"""
    print("\n" + "="*50)
    print("Multivariate Analysis")
    print("="*50)
    
    # Multivariate: Pair plot
    numeric_cols = ['Valuation_B', 'Total_Raised_B', 'Investors Count', 'Years_to_Unicorn']
    valid_data = df[numeric_cols].dropna()
    
    if len(valid_data) > 0:
        sns.pairplot(valid_data, diag_kind='kde', plot_kws={'alpha': 0.6})
        plt.suptitle('Pair Plot of Key Variables', y=1.02)
        plt.tight_layout()
        plt.savefig('../../results/plots/multivariate_pairplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: multivariate_pairplot.png")
    
    # Multivariate: Heatmap with categorical grouping
    top_countries = df['Country'].value_counts().head(5).index
    top_industries = df['Industry'].value_counts().head(5).index
    
    filtered_df = df[(df['Country'].isin(top_countries)) & (df['Industry'].isin(top_industries))]
    pivot_table = filtered_df.pivot_table(values='Valuation_B', index='Country', 
                                          columns='Industry', aggfunc='mean')
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='YlOrRd', 
                cbar_kws={'label': 'Mean Valuation ($B)'})
    plt.title('Mean Valuation by Country and Industry')
    plt.tight_layout()
    plt.savefig('../../results/plots/multivariate_heatmap_country_industry.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: multivariate_heatmap_country_industry.png")
    
    # Multivariate: 3D relationship
    valid_data_3d = df[['Total_Raised_B', 'Investors Count', 'Valuation_B']].dropna()
    if len(valid_data_3d) > 0:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(valid_data_3d['Total_Raised_B'], 
                            valid_data_3d['Investors Count'], 
                            valid_data_3d['Valuation_B'], 
                            c=valid_data_3d['Valuation_B'], cmap='viridis', alpha=0.6, s=50)
        
        ax.set_xlabel('Total Raised ($B)')
        ax.set_ylabel('Investors Count')
        ax.set_zlabel('Valuation ($B)')
        ax.set_title('3D Scatter: Valuation, Total Raised, and Investors Count')
        
        plt.colorbar(scatter, label='Valuation ($B)')
        plt.tight_layout()
        plt.savefig('../../results/plots/multivariate_3d_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: multivariate_3d_scatter.png")
    
    # Multivariate: Violin plot
    top_3_countries = df['Country'].value_counts().head(3).index
    filtered_df2 = df[df['Country'].isin(top_3_countries)]
    
    plt.figure(figsize=(14, 8))
    sns.violinplot(data=filtered_df2, x='Country', y='Valuation_B', hue='Financial Stage', split=True)
    plt.title('Valuation Distribution by Country and Financial Stage')
    plt.xlabel('Country')
    plt.ylabel('Valuation ($B)')
    plt.xticks(rotation=45)
    plt.legend(title='Financial Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('../../results/plots/multivariate_violin.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: multivariate_violin.png")
    
    # Summary statistics for multivariate analysis
    print("\nMultivariate Summary:")
    print("\nCorrelation Matrix:")
    corr_cols = ['Valuation_B', 'Total_Raised_B', 'Investors Count', 'Deal Terms', 
                 'Portfolio Exits', 'Years_to_Unicorn']
    corr_matrix = df[corr_cols].corr()
    print(corr_matrix)
    
    print("\n\nMean Valuation by Country (Top 10):")
    print(df.groupby('Country')['Valuation_B'].mean().sort_values(ascending=False).head(10))
    
    print("\n\nMean Valuation by Industry (Top 10):")
    print(df.groupby('Industry')['Valuation_B'].mean().sort_values(ascending=False).head(10))


def main():
    """Main function"""
    print("="*50)
    print("Univariate, Bivariate, and Multivariate Analysis")
    print("="*50)
    
    # Load data
    df = load_cleaned_data()
    
    # Perform analyses
    univariate_analysis(df)
    bivariate_analysis(df)
    multivariate_analysis(df)
    
    print("\n" + "="*50)
    print("Analysis Completed Successfully!")
    print("="*50)


if __name__ == "__main__":
    main()


