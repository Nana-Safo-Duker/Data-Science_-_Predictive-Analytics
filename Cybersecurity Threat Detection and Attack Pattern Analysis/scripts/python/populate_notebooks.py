"""
Script to populate all analysis notebooks with comprehensive content
"""
import json
import os

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else [source]
    }

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [source]
    }

# Define notebook contents
notebook_contents = {
    "notebooks/python/03_Univariate_Bivariate_Multivariate_Analysis.ipynb": {
        "cells": [
            create_markdown_cell("# Univariate, Bivariate, and Multivariate Analysis\n\nThis notebook provides comprehensive analysis of individual variables, relationships between variables, and patterns across multiple variables."),
            create_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom scipy import stats\nimport warnings\nwarnings.filterwarnings('ignore')\n\nsns.set_style('whitegrid')\nplt.rcParams['figure.figsize'] = (14, 8)\npd.set_option('display.max_columns', None)"),
            create_markdown_cell("## 1. Data Loading"),
            create_code_cell("# Load data\ndf = pd.read_csv('../../data/Cybersecurity_attacks.csv')\ndf.columns = df.columns.str.strip()\nif '.' in df.columns:\n    df = df.drop(columns=['.'])\n\n# Parse Time\nif 'Time' in df.columns:\n    def parse_time(time_str):\n        if pd.isna(time_str):\n            return None, None\n        try:\n            if '-' in str(time_str):\n                start, end = str(time_str).split('-')\n                return int(start), int(end)\n            else:\n                return int(time_str), int(time_str)\n        except:\n            return None, None\n    time_parsed = df['Time'].apply(parse_time)\n    df['Time_Start'] = [t[0] for t in time_parsed]\n    df['Time_End'] = [t[1] for t in time_parsed]\n    df['Time_Duration'] = df['Time_End'] - df['Time_Start']\n    df['Datetime_Start'] = pd.to_datetime(df['Time_Start'], unit='s', errors='coerce')\n    df['Hour'] = df['Datetime_Start'].dt.hour\n    df['DayOfWeek'] = df['Datetime_Start'].dt.day_name()\n\nprint(f\"Dataset Shape: {df.shape}\")"),
            create_markdown_cell("## 2. Univariate Analysis"),
            create_code_cell("# Numerical variables\nnumerical_cols = ['Source Port', 'Destination Port', 'Time_Duration', 'Hour']\nnumerical_cols = [col for col in numerical_cols if col in df.columns]\n\nfor col in numerical_cols:\n    fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n    df[col].hist(bins=50, ax=axes[0, 0], edgecolor='black')\n    axes[0, 0].set_title(f'{col} - Histogram')\n    df.boxplot(column=col, ax=axes[0, 1])\n    axes[0, 1].set_title(f'{col} - Box Plot')\n    stats.probplot(df[col].dropna(), dist='norm', plot=axes[1, 0])\n    axes[1, 0].set_title(f'{col} - Q-Q Plot')\n    axes[1, 1].axis('off')\n    stats_text = f\"Mean: {df[col].mean():.2f}\\nMedian: {df[col].median():.2f}\\nStd: {df[col].std():.2f}\\nSkew: {df[col].skew():.4f}\"\n    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')\n    plt.suptitle(f'Univariate Analysis: {col}', fontsize=16, fontweight='bold')\n    plt.tight_layout()\n    plt.savefig(f'../../visualizations/univariate_{col.replace(\" \", \"_\")}.png', dpi=300, bbox_inches='tight')\n    plt.show()"),
            create_markdown_cell("## 3. Bivariate Analysis"),
            create_code_cell("# Scatter plot: Source Port vs Destination Port\nif 'Source Port' in df.columns and 'Destination Port' in df.columns:\n    sample_df = df.sample(min(10000, len(df)), random_state=42)\n    plt.figure(figsize=(12, 8))\n    plt.scatter(sample_df['Source Port'], sample_df['Destination Port'], alpha=0.5, s=10)\n    plt.xlabel('Source Port')\n    plt.ylabel('Destination Port')\n    plt.title('Source Port vs Destination Port', fontsize=16, fontweight='bold')\n    plt.grid(True, alpha=0.3)\n    plt.tight_layout()\n    plt.savefig('../../visualizations/bivariate_source_dest_port.png', dpi=300, bbox_inches='tight')\n    plt.show()\n    correlation = df['Source Port'].corr(df['Destination Port'])\n    print(f\"Correlation: {correlation:.4f}\")"),
            create_code_cell("# Box plot: Destination Port by Attack Category\nif 'Attack category' in df.columns and 'Destination Port' in df.columns:\n    top_categories = df['Attack category'].value_counts().head(5).index\n    filtered_df = df[df['Attack category'].isin(top_categories)]\n    plt.figure(figsize=(14, 8))\n    sns.boxplot(data=filtered_df, x='Attack category', y='Destination Port')\n    plt.title('Destination Port by Attack Category', fontsize=16, fontweight='bold')\n    plt.xlabel('Attack Category')\n    plt.ylabel('Destination Port')\n    plt.xticks(rotation=45, ha='right')\n    plt.tight_layout()\n    plt.savefig('../../visualizations/bivariate_category_port.png', dpi=300, bbox_inches='tight')\n    plt.show()"),
            create_markdown_cell("## 4. Multivariate Analysis"),
            create_code_cell("# Correlation matrix\ncorrelation_matrix = df[numerical_cols].corr()\nplt.figure(figsize=(12, 10))\nsns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt='.3f')\nplt.title('Multivariate Correlation Matrix', fontsize=16, fontweight='bold')\nplt.tight_layout()\nplt.savefig('../../visualizations/multivariate_correlation.png', dpi=300, bbox_inches='tight')\nplt.show()\nprint(\"Correlation Matrix:\")\nprint(correlation_matrix.round(4))"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.8.0"}
        }
    }
}

# Populate notebooks
for notebook_path, notebook_data in notebook_contents.items():
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
    notebook = {
        "cells": notebook_data["cells"],
        "metadata": notebook_data["metadata"],
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Populated: {notebook_path}")

print("Notebook population complete!")



