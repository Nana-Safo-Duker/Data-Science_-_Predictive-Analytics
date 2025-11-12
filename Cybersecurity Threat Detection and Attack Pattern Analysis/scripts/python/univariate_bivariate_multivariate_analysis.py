"""
Univariate, Bivariate, and Multivariate Analysis for Cybersecurity Attacks Dataset
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path):
    """Load and prepare data"""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    if '.' in df.columns:
        df = df.drop(columns=['.'])
    
    # Parse Time column
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
        df['Datetime_Start'] = pd.to_datetime(df['Time_Start'], unit='s', errors='coerce')
        df['Hour'] = df['Datetime_Start'].dt.hour
        df['DayOfWeek'] = df['Datetime_Start'].dt.day_name()
        df['Month'] = df['Datetime_Start'].dt.month
    
    return df

def univariate_analysis(df, numerical_cols, categorical_cols, output_dir='../../visualizations/'):
    """Perform univariate analysis"""
    print("=" * 60)
    print("UNIVARIATE ANALYSIS")
    print("=" * 60)
    
    # Numerical variables
    for col in numerical_cols:
        if col not in df.columns:
            continue
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Histogram
        df[col].hist(bins=50, ax=axes[0, 0], edgecolor='black')
        axes[0, 0].set_title(f'{col} - Histogram', fontweight='bold')
        axes[0, 0].set_xlabel(col)
        axes[0, 0].set_ylabel('Frequency')
        
        # Box plot
        df.boxplot(column=col, ax=axes[0, 1])
        axes[0, 1].set_title(f'{col} - Box Plot', fontweight='bold')
        axes[0, 1].set_ylabel(col)
        
        # Q-Q plot
        stats.probplot(df[col].dropna(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title(f'{col} - Q-Q Plot', fontweight='bold')
        
        # Summary statistics
        axes[1, 1].axis('off')
        stats_text = f"Summary Statistics\n\n"
        stats_text += f"Mean: {df[col].mean():.2f}\n"
        stats_text += f"Median: {df[col].median():.2f}\n"
        stats_text += f"Std Dev: {df[col].std():.2f}\n"
        stats_text += f"Skewness: {df[col].skew():.4f}\n"
        stats_text += f"Kurtosis: {df[col].kurtosis():.4f}\n"
        stats_text += f"Min: {df[col].min():.2f}\n"
        stats_text += f"Max: {df[col].max():.2f}"
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                        family='monospace')
        
        plt.suptitle(f'Univariate Analysis: {col}', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{output_dir}univariate_{col.replace(" ", "_")}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n{col} Statistics:")
        print(df[col].describe())

def bivariate_analysis(df, output_dir='../../visualizations/'):
    """Perform bivariate analysis"""
    print("\n" + "=" * 60)
    print("BIVARIATE ANALYSIS")
    print("=" * 60)
    
    # Scatter plot: Source Port vs Destination Port
    if 'Source Port' in df.columns and 'Destination Port' in df.columns:
        sample_df = df.sample(min(10000, len(df)), random_state=42)
        plt.figure(figsize=(12, 8))
        plt.scatter(sample_df['Source Port'], sample_df['Destination Port'], 
                    alpha=0.5, s=10)
        plt.xlabel('Source Port', fontsize=12)
        plt.ylabel('Destination Port', fontsize=12)
        plt.title('Source Port vs Destination Port', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}bivariate_source_dest_port.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        correlation = df['Source Port'].corr(df['Destination Port'])
        print(f"\nCorrelation between Source Port and Destination Port: {correlation:.4f}")
    
    # Box plot: Destination Port by Attack Category
    if 'Attack category' in df.columns and 'Destination Port' in df.columns:
        top_categories = df['Attack category'].value_counts().head(5).index
        filtered_df = df[df['Attack category'].isin(top_categories)]
        
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=filtered_df, x='Attack category', y='Destination Port')
        plt.title('Destination Port by Attack Category', fontsize=16, fontweight='bold')
        plt.xlabel('Attack Category', fontsize=12)
        plt.ylabel('Destination Port', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{output_dir}bivariate_category_port.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

def multivariate_analysis(df, numerical_cols, output_dir='../../visualizations/'):
    """Perform multivariate analysis"""
    print("\n" + "=" * 60)
    print("MULTIVARIATE ANALYSIS")
    print("=" * 60)
    
    # Correlation matrix
    numerical_cols_available = [col for col in numerical_cols if col in df.columns]
    if len(numerical_cols_available) > 1:
        correlation_matrix = df[numerical_cols_available].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=1, cbar_kws={'shrink': 0.8}, fmt='.3f')
        plt.title('Multivariate Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}multivariate_correlation.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nCorrelation Matrix:")
        print(correlation_matrix.round(4))

def main():
    """Main function"""
    # Load data
    df = load_and_prepare_data('../../data/Cybersecurity_attacks.csv')
    
    # Define columns
    numerical_cols = ['Source Port', 'Destination Port', 'Time_Duration', 'Hour', 'Month']
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    categorical_cols = ['Attack category', 'Protocol', 'Attack subcategory']
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    # Perform analyses
    univariate_analysis(df, numerical_cols, categorical_cols)
    bivariate_analysis(df)
    multivariate_analysis(df, numerical_cols)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()



