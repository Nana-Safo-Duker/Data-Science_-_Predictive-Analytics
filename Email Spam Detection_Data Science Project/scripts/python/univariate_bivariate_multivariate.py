"""
Univariate, Bivariate, and Multivariate Analysis
Email Spam Detection Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_data():
    """Load processed dataset"""
    try:
        df = pd.read_csv('../../data/emails_spam_processed.csv')
    except:
        df = pd.read_csv('../../data/emails_spam_clean.csv')
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
    return df

def univariate_analysis(df):
    """Perform univariate analysis"""
    print("="*60)
    print("UNIVARIATE ANALYSIS")
    print("="*60)
    
    numeric_cols = ['text_length', 'word_count', 'sentence_count', 'avg_word_length']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    # Analyze each variable independently
    for col in numeric_cols:
        print(f"\n1. Analysis of {col}:")
        print(f"   Mean: {df[col].mean():.2f}")
        print(f"   Median: {df[col].median():.2f}")
        print(f"   Mode: {df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A'}")
        print(f"   Std Dev: {df[col].std():.2f}")
        print(f"   Skewness: {df[col].skew():.2f}")
        print(f"   Kurtosis: {df[col].kurtosis():.2f}")
        print(f"   Min: {df[col].min():.2f}")
        print(f"   Max: {df[col].max():.2f}")
        print(f"   Range: {df[col].max() - df[col].min():.2f}")
        print(f"   Q1: {df[col].quantile(0.25):.2f}")
        print(f"   Q2 (Median): {df[col].quantile(0.50):.2f}")
        print(f"   Q3: {df[col].quantile(0.75):.2f}")
        print(f"   IQR: {df[col].quantile(0.75) - df[col].quantile(0.25):.2f}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for idx, col in enumerate(numeric_cols[:4]):
        # Histogram
        axes[idx].hist(df[col], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[idx].axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean: {df[col].mean():.2f}')
        axes[idx].axvline(df[col].median(), color='green', linestyle='--', label=f'Median: {df[col].median():.2f}')
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../output/figures/univariate_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Box plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for idx, col in enumerate(numeric_cols[:4]):
        df.boxplot(column=col, ax=axes[idx])
        axes[idx].set_title(f'Box Plot of {col}')
        axes[idx].set_ylabel(col)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../output/figures/univariate_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()

def bivariate_analysis(df):
    """Perform bivariate analysis"""
    print("\n" + "="*60)
    print("BIVARIATE ANALYSIS")
    print("="*60)
    
    numeric_cols = ['text_length', 'word_count', 'sentence_count', 'avg_word_length']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    # Analyze relationship between each feature and target variable
    print("\n1. Relationship between Features and Target Variable (Spam):")
    for col in numeric_cols:
        spam_mean = df[df['spam']==1][col].mean()
        ham_mean = df[df['spam']==0][col].mean()
        difference = spam_mean - ham_mean
        percent_diff = (difference / ham_mean) * 100 if ham_mean != 0 else 0
        
        print(f"\n{col}:")
        print(f"   Spam Mean: {spam_mean:.2f}")
        print(f"   Ham Mean: {ham_mean:.2f}")
        print(f"   Difference: {difference:.2f} ({percent_diff:.2f}%)")
        
        # T-test
        t_stat, p_value = stats.ttest_ind(df[df['spam']==1][col], df[df['spam']==0][col])
        print(f"   T-test p-value: {p_value:.6f}")
        print(f"   Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for idx, col in enumerate(numeric_cols[:4]):
        axes[idx].scatter(df[df['spam']==0][col], df[df['spam']==0]['spam'], 
                         alpha=0.5, label='Ham', color='skyblue', s=10)
        axes[idx].scatter(df[df['spam']==1][col], df[df['spam']==1]['spam'], 
                         alpha=0.5, label='Spam', color='salmon', s=10)
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Spam (0=Ham, 1=Spam)')
        axes[idx].set_title(f'{col} vs Spam')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../output/figures/bivariate_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Violin plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for idx, col in enumerate(numeric_cols[:4]):
        data_to_plot = [df[df['spam']==0][col], df[df['spam']==1][col]]
        axes[idx].violinplot(data_to_plot, positions=[0, 1], showmeans=True)
        axes[idx].set_xticks([0, 1])
        axes[idx].set_xticklabels(['Ham', 'Spam'])
        axes[idx].set_ylabel(col)
        axes[idx].set_title(f'Distribution of {col} by Spam/Ham')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../output/figures/bivariate_violin.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Correlation with target
    print("\n2. Correlation with Target Variable:")
    for col in numeric_cols:
        correlation = df[col].corr(df['spam'])
        print(f"   {col} - Spam correlation: {correlation:.4f}")

def multivariate_analysis(df):
    """Perform multivariate analysis"""
    print("\n" + "="*60)
    print("MULTIVARIATE ANALYSIS")
    print("="*60)
    
    numeric_cols = ['text_length', 'word_count', 'sentence_count', 'avg_word_length']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    # Correlation matrix
    print("\n1. Correlation Matrix:")
    correlation_matrix = df[numeric_cols + ['spam']].corr()
    print(correlation_matrix)
    
    # Visualize correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.3f')
    plt.title('Correlation Matrix - Multivariate Analysis')
    plt.tight_layout()
    plt.savefig('../../output/figures/multivariate_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Pair plots
    print("\n2. Pairwise Relationships:")
    # Sample data for pair plot (too large otherwise)
    sample_df = df.sample(min(1000, len(df)), random_state=42)
    
    pair_plot_cols = numeric_cols[:4] + ['spam']
    pair_df = sample_df[pair_plot_cols].copy()
    pair_df['spam'] = pair_df['spam'].astype(str)
    
    # Create pair plot
    g = sns.pairplot(pair_df, hue='spam', diag_kind='kde', 
                     palette={'0': 'skyblue', '1': 'salmon'})
    g.fig.suptitle('Pairwise Relationships - Multivariate Analysis', y=1.02)
    plt.savefig('../../output/figures/multivariate_pairplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Principal Component Analysis (PCA) visualization
    print("\n3. Principal Component Analysis:")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data
    X = df[numeric_cols].fillna(0)
    y = df['spam']
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Visualize PCA
    plt.figure(figsize=(12, 8))
    plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], alpha=0.5, label='Ham', color='skyblue', s=10)
    plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], alpha=0.5, label='Spam', color='salmon', s=10)
    plt.xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})')
    plt.title('PCA - Multivariate Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../../output/figures/multivariate_pca.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Explained variance by PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"   Explained variance by PC2: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"   Total explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Feature interactions
    print("\n4. Feature Interactions:")
    if 'text_length' in df.columns and 'word_count' in df.columns:
        # Create interaction term
        df['text_length_word_count_interaction'] = df['text_length'] * df['word_count']
        interaction_corr = df['text_length_word_count_interaction'].corr(df['spam'])
        print(f"   Text Length × Word Count interaction - Spam correlation: {interaction_corr:.4f}")

def main():
    """Main function"""
    print("="*60)
    print("UNIVARIATE, BIVARIATE, AND MULTIVARIATE ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Univariate analysis
    univariate_analysis(df)
    
    # Bivariate analysis
    bivariate_analysis(df)
    
    # Multivariate analysis
    multivariate_analysis(df)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)

if __name__ == "__main__":
    main()


