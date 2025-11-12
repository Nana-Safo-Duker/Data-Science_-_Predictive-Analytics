"""
Script to create comprehensive analysis notebooks for Cybersecurity Attacks dataset
"""
import json
import os

def create_cell(cell_type, source, metadata=None):
    """Create a notebook cell"""
    cell = {
        "cell_type": cell_type,
        "metadata": metadata or {},
        "source": source if isinstance(source, list) else [source]
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

def create_statistical_analysis_notebook():
    """Create Statistical Analysis notebook"""
    cells = [
        create_cell("markdown", "# Statistical Analysis - Cybersecurity Attacks Dataset\n\n## Overview\nThis notebook provides comprehensive statistical analysis including:\n1. Descriptive Statistics\n2. Inferential Statistics\n3. Exploratory Statistical Analysis\n\n## Objectives\n- Compute descriptive statistics for all variables\n- Perform hypothesis testing\n- Conduct inferential statistical analysis\n- Explore statistical relationships and patterns"),
        create_cell("code", "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom scipy import stats\nfrom scipy.stats import chi2_contingency, f_oneway, ttest_ind, mannwhitneyu\nfrom scipy.stats import normaltest, shapiro, kstest\nimport warnings\nwarnings.filterwarnings('ignore')\n\nsns.set_style('whitegrid')\nplt.rcParams['figure.figsize'] = (12, 6)\npd.set_option('display.max_columns', None)"),
        create_cell("markdown", "## 1. Data Loading and Preparation"),
        create_cell("code", "# Load and prepare data\ndf = pd.read_csv('../../data/Cybersecurity_attacks.csv')\ndf.columns = df.columns.str.strip()\nif '.' in df.columns:\n    df = df.drop(columns=['.'])\n\n# Parse Time column\nif 'Time' in df.columns:\n    def parse_time(time_str):\n        if pd.isna(time_str):\n            return None, None\n        try:\n            if '-' in str(time_str):\n                start, end = str(time_str).split('-')\n                return int(start), int(end)\n            else:\n                return int(time_str), int(time_str)\n        except:\n            return None, None\n    \n    time_parsed = df['Time'].apply(parse_time)\n    df['Time_Start'] = [t[0] for t in time_parsed]\n    df['Time_End'] = [t[1] for t in time_parsed]\n    df['Time_Duration'] = df['Time_End'] - df['Time_Start']\n    df['Datetime_Start'] = pd.to_datetime(df['Time_Start'], unit='s', errors='coerce')\n    df['Hour'] = df['Datetime_Start'].dt.hour\n    df['DayOfWeek'] = df['Datetime_Start'].dt.day_name()\n\nprint(f\"Dataset Shape: {df.shape}\")"),
        create_cell("markdown", "## 2. Descriptive Statistics"),
        create_cell("code", "# Descriptive Statistics for Numerical Variables\nnumerical_cols = ['Source Port', 'Destination Port', 'Time_Duration', 'Hour']\nnumerical_cols = [col for col in numerical_cols if col in df.columns]\n\nprint(\"DESCRIPTIVE STATISTICS\")\nfor col in numerical_cols:\n    print(f\"\\n{col}:\")\n    print(df[col].describe())\n    print(f\"Skewness: {df[col].skew():.4f}\")\n    print(f\"Kurtosis: {df[col].kurtosis():.4f}\")"),
        create_cell("markdown", "## 3. Hypothesis Testing"),
        create_cell("code", "# Chi-square test for independence\nif 'Attack category' in df.columns and 'Protocol' in df.columns:\n    top_categories = df['Attack category'].value_counts().head(5).index\n    top_protocols = df['Protocol'].value_counts().head(5).index\n    contingency_table = pd.crosstab(\n        df[df['Attack category'].isin(top_categories)]['Attack category'],\n        df[df['Protocol'].isin(top_protocols)]['Protocol']\n    )\n    chi2, p_value, dof, expected = chi2_contingency(contingency_table)\n    print(f\"Chi-square: {chi2:.4f}, p-value: {p_value:.6f}\")\n    print(f\"Conclusion: {'Dependent' if p_value < 0.05 else 'Independent'}\")"),
        create_cell("code", "# ANOVA test\nif 'Attack category' in df.columns and 'Destination Port' in df.columns:\n    top_categories = df['Attack category'].value_counts().head(5).index.tolist()\n    filtered_df = df[df['Attack category'].isin(top_categories)]\n    groups = [filtered_df[filtered_df['Attack category'] == cat]['Destination Port'].dropna().values \n              for cat in top_categories]\n    f_stat, p_value = f_oneway(*groups)\n    print(f\"F-statistic: {f_stat:.4f}, p-value: {p_value:.6f}\")\n    print(f\"Conclusion: {'Means differ' if p_value < 0.05 else 'Means are similar'}\")"),
        create_cell("markdown", "## 4. Correlation Analysis"),
        create_cell("code", "# Correlation matrix\ncorrelation_matrix = df[numerical_cols].corr()\nplt.figure(figsize=(10, 8))\nsns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)\nplt.title('Correlation Matrix', fontsize=16, fontweight='bold')\nplt.tight_layout()\nplt.savefig('../../visualizations/correlation_matrix.png', dpi=300, bbox_inches='tight')\nplt.show()")
    ]
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.8.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    return notebook

# Create all notebooks
notebooks_to_create = [
    ("notebooks/python/02_Statistical_Analysis.ipynb", create_statistical_analysis_notebook()),
]

for path, notebook in notebooks_to_create:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Created: {path}")



