"""
Univariate, Bivariate, and Multivariate Analysis
Customer Data Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, pearsonr
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

# Set paths
data_path = Path('../../data/Customers.csv')
results_path = Path('../../results')
results_path.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(data_path)

def univariate_analysis():
    """Perform univariate analysis"""
    print("=" * 50)
    print("UNIVARIATE ANALYSIS")
    print("=" * 50)
    
    # Numerical variables
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\n=== Numerical Variables Univariate Analysis ===")
    for col in numerical_cols:
        print(f"\n{col}:")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Median: {df[col].median():.2f}")
        print(f"  Mode: {df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A'}")
        print(f"  Std Dev: {df[col].std():.2f}")
        print(f"  Min: {df[col].min()}")
        print(f"  Max: {df[col].max()}")
        print(f"  Range: {df[col].max() - df[col].min()}")
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Histogram
        axes[0].hist(df[col], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Histogram of {col}')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(df[col], vert=True)
        axes[1].set_ylabel(col)
        axes[1].set_title(f'Box Plot of {col}')
        axes[1].grid(True, alpha=0.3)
        
        # Violin plot
        axes[2].violinplot([df[col]], vert=True)
        axes[2].set_ylabel(col)
        axes[2].set_title(f'Violin Plot of {col}')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_path / f'univariate_{col}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print("\n=== Categorical Variables Univariate Analysis ===")
    for col in categorical_cols:
        print(f"\n{col}:")
        value_counts = df[col].value_counts()
        print(f"  Total categories: {df[col].nunique()}")
        print(f"  Most frequent: {value_counts.index[0]} ({value_counts.iloc[0]} occurrences)")
        print(f"  Frequency percentage: {(value_counts.iloc[0] / len(df)) * 100:.2f}%")
        
        # Visualization
        plt.figure(figsize=(12, 6))
        if df[col].nunique() > 20:
            # Show top 20
            value_counts.head(20).plot(kind='bar', color='coral', edgecolor='black')
            plt.title(f'Top 20 {col} Distribution')
        else:
            value_counts.plot(kind='bar', color='coral', edgecolor='black')
            plt.title(f'{col} Distribution')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_path / f'univariate_{col}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\n✓ Univariate analysis visualizations saved")

def bivariate_analysis():
    """Perform bivariate analysis"""
    print("\n" + "=" * 50)
    print("BIVARIATE ANALYSIS")
    print("=" * 50)
    
    # Country vs City
    print("\n=== Country vs City Analysis ===")
    country_city = pd.crosstab(df['Country'], df['City'])
    print(f"Contingency table shape: {country_city.shape}")
    print("\nTop country-city combinations:")
    country_city_flat = country_city.stack().reset_index()
    country_city_flat.columns = ['Country', 'City', 'Count']
    country_city_flat = country_city_flat[country_city_flat['Count'] > 0].sort_values('Count', ascending=False)
    print(country_city_flat.head(10))
    
    # Visualization
    plt.figure(figsize=(16, 10))
    sns.heatmap(country_city, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Count'})
    plt.title('Country vs City Heatmap')
    plt.xlabel('City')
    plt.ylabel('Country')
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(results_path / 'bivariate_country_city.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Customer ID vs Country
    print("\n=== Customer ID Distribution by Country ===")
    country_customer = df.groupby('Country')['CustomerID'].count().sort_values(ascending=False)
    print(country_customer)
    
    plt.figure(figsize=(14, 8))
    country_customer.plot(kind='bar', color='steelblue', edgecolor='black')
    plt.xlabel('Country')
    plt.ylabel('Number of Customers')
    plt.title('Customer Distribution by Country')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_path / 'bivariate_customer_country.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chi-square test for Country and City
    print("\n=== Chi-Square Test: Country vs City ===")
    contingency_table = pd.crosstab(df['Country'], df['City'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Degrees of freedom: {dof}")
    
    alpha = 0.05
    if p_value < alpha:
        print(f"Result: Significant association between Country and City (p < {alpha})")
    else:
        print(f"Result: No significant association (p >= {alpha})")
    
    print("\n✓ Bivariate analysis visualizations saved")

def multivariate_analysis():
    """Perform multivariate analysis"""
    print("\n" + "=" * 50)
    print("MULTIVARIATE ANALYSIS")
    print("=" * 50)
    
    # Encode categorical variables for multivariate analysis
    le_country = LabelEncoder()
    le_city = LabelEncoder()
    
    df_encoded = df.copy()
    df_encoded['Country_encoded'] = le_country.fit_transform(df['Country'])
    df_encoded['City_encoded'] = le_city.fit_transform(df['City'])
    
    # Select features for multivariate analysis
    features = ['CustomerID', 'Country_encoded', 'City_encoded']
    X = df_encoded[features]
    
    # Correlation matrix
    print("\n=== Correlation Matrix ===")
    corr_matrix = X.corr()
    print(corr_matrix)
    
    # Visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix - Multivariate Analysis')
    plt.tight_layout()
    plt.savefig(results_path / 'multivariate_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # PCA Analysis
    print("\n=== Principal Component Analysis (PCA) ===")
    pca = PCA(n_components=min(3, len(features)))
    X_scaled = (X - X.mean()) / X.std()
    pca_result = pca.fit_transform(X_scaled)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
    
    # PCA Visualization
    if pca_result.shape[1] >= 2:
        plt.figure(figsize=(12, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, c=df_encoded['Country_encoded'], cmap='viridis')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA: First Two Principal Components')
        plt.colorbar(label='Country (encoded)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_path / 'multivariate_pca.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Country-City-CustomerID relationship
    print("\n=== Country-City-CustomerID Relationship ===")
    country_city_summary = df.groupby(['Country', 'City']).agg({
        'CustomerID': ['count', 'min', 'max']
    }).reset_index()
    country_city_summary.columns = ['Country', 'City', 'Customer_Count', 'Min_CustomerID', 'Max_CustomerID']
    print(country_city_summary.head(10))
    
    # 3D visualization (if applicable)
    if len(features) >= 3:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(df_encoded['CustomerID'], 
                            df_encoded['Country_encoded'], 
                            df_encoded['City_encoded'],
                            c=df_encoded['Country_encoded'], 
                            cmap='viridis', 
                            alpha=0.6)
        ax.set_xlabel('Customer ID')
        ax.set_ylabel('Country (encoded)')
        ax.set_zlabel('City (encoded)')
        ax.set_title('3D Scatter Plot: Customer ID vs Country vs City')
        plt.colorbar(scatter, ax=ax, label='Country (encoded)')
        plt.tight_layout()
        plt.savefig(results_path / 'multivariate_3d_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\n✓ Multivariate analysis visualizations saved")

def main():
    """Main function"""
    univariate_analysis()
    bivariate_analysis()
    multivariate_analysis()
    
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    print("\n1. Univariate Analysis:")
    print("   • Analyzed individual variables (numerical and categorical)")
    print("   • Generated distribution plots and summary statistics")
    print("\n2. Bivariate Analysis:")
    print("   • Examined relationships between two variables")
    print("   • Performed chi-square tests for independence")
    print("   • Created contingency tables and heatmaps")
    print("\n3. Multivariate Analysis:")
    print("   • Analyzed relationships among multiple variables")
    print("   • Performed PCA for dimensionality reduction")
    print("   • Created correlation matrices and 3D visualizations")

if __name__ == "__main__":
    main()

