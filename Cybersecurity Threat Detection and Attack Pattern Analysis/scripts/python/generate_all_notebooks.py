"""
Comprehensive script to generate all analysis notebooks with full content
"""
import json
import os

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else [line + "\n" for line in source.split("\n")]
    }

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [line + "\n" for line in source.split("\n")]
    }

# Common data loading code
data_loading_code = """# Load and prepare data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('../../data/Cybersecurity_attacks.csv')
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

print(f"Dataset Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")"""

# Define all notebooks
notebooks = {
    "notebooks/python/02_Statistical_Analysis.ipynb": {
        "title": "Statistical Analysis - Cybersecurity Attacks Dataset",
        "cells": [
            create_markdown_cell("# Statistical Analysis - Cybersecurity Attacks Dataset\n\n## Overview\nThis notebook provides comprehensive statistical analysis including descriptive statistics, inferential statistics, and exploratory statistical analysis."),
            create_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, mannwhitneyu, normaltest
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
pd.set_option('display.max_columns', None)"""),
            create_markdown_cell("## 1. Data Loading and Preparation"),
            create_code_cell(data_loading_code),
            create_markdown_cell("## 2. Descriptive Statistics"),
            create_code_cell("""# Descriptive Statistics for Numerical Variables
numerical_cols = ['Source Port', 'Destination Port', 'Time_Duration', 'Hour']
numerical_cols = [col for col in numerical_cols if col in df.columns]

print("=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

for col in numerical_cols:
    print(f"\\n{col}:")
    print(df[col].describe())
    print(f"Skewness: {df[col].skew():.4f}")
    print(f"Kurtosis: {df[col].kurtosis():.4f}")"""),
            create_markdown_cell("## 3. Hypothesis Testing"),
            create_code_cell("""# Chi-square test for independence
if 'Attack category' in df.columns and 'Protocol' in df.columns:
    top_categories = df['Attack category'].value_counts().head(5).index
    top_protocols = df['Protocol'].value_counts().head(5).index
    contingency_table = pd.crosstab(
        df[df['Attack category'].isin(top_categories)]['Attack category'],
        df[df['Protocol'].isin(top_protocols)]['Protocol']
    )
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-square: {chi2:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Conclusion: {'Dependent' if p_value < 0.05 else 'Independent'}")"""),
            create_code_cell("""# ANOVA test
if 'Attack category' in df.columns and 'Destination Port' in df.columns:
    top_categories = df['Attack category'].value_counts().head(5).index.tolist()
    filtered_df = df[df['Attack category'].isin(top_categories)]
    groups = [filtered_df[filtered_df['Attack category'] == cat]['Destination Port'].dropna().values 
              for cat in top_categories]
    f_stat, p_value = f_oneway(*groups)
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Conclusion: {'Means differ' if p_value < 0.05 else 'Means are similar'}")"""),
            create_markdown_cell("## 4. Correlation Analysis"),
            create_code_cell("""# Correlation matrix
correlation_matrix = df[numerical_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../../visualizations/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("Correlation Matrix:")
print(correlation_matrix.round(4))"""),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.8.0"}
        }
    },
    "notebooks/python/04_ML_Analysis.ipynb": {
        "title": "Machine Learning Analysis - Cybersecurity Attacks Dataset",
        "cells": [
            create_markdown_cell("# Machine Learning Analysis - Cybersecurity Attacks Dataset\n\n## Overview\nThis notebook implements multiple machine learning algorithms to classify cybersecurity attacks."),
            create_code_cell("""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')"""),
            create_markdown_cell("## 1. Data Loading and Preparation"),
            create_code_cell(data_loading_code),
            create_markdown_cell("## 2. Feature Engineering"),
            create_code_cell("""# Prepare features
features = ['Source Port', 'Destination Port', 'Hour', 'Month']
if 'Time_Duration' in df.columns:
    features.append('Time_Duration')

# Encode categorical variables
le_protocol = LabelEncoder()
le_category = LabelEncoder()

if 'Protocol' in df.columns:
    df['Protocol_encoded'] = le_protocol.fit_transform(df['Protocol'].astype(str))
    features.append('Protocol_encoded')

if 'DayOfWeek' in df.columns:
    day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 
                   'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    df['DayOfWeek_encoded'] = df['DayOfWeek'].map(day_mapping).fillna(0)
    features.append('DayOfWeek_encoded')

# Target variable
if 'Attack category' in df.columns:
    df['Attack_category_encoded'] = le_category.fit_transform(df['Attack category'].astype(str))
    target = 'Attack_category_encoded'
else:
    target = None

features = [f for f in features if f in df.columns]
df[features] = df[features].fillna(df[features].median())

X = df[features]
y = df[target] if target else None

print(f"Features: {features}")
print(f"Target classes: {len(le_category.classes_) if target else 0}")
print(f"Dataset shape: {X.shape}")"""),
            create_markdown_cell("## 3. Model Training"),
            create_code_cell("""# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
    'SVM': SVC(random_state=42, probability=True),
    'Naive Bayes': GaussianNB()
}

results = {}

for name, model in models.items():
    print(f"\\nTraining {name}...")
    
    if name in ['SVM', 'Logistic Regression']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")"""),
            create_markdown_cell("## 4. Model Evaluation"),
            create_code_cell("""# Print results summary
print("\\n" + "="*80)
print("MODEL PERFORMANCE SUMMARY")
print("="*80)
print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-"*80)

for name, result in results.items():
    print(f"{name:<20} {result['accuracy']:<12.4f} {result['precision']:<12.4f} "
          f"{result['recall']:<12.4f} {result['f1']:<12.4f}")

# Find best model
best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
print(f"\\nBest Model: {best_model_name} (F1-Score: {results[best_model_name]['f1']:.4f})")"""),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.8.0"}
        }
    }
}

# Generate notebooks
for notebook_path, notebook_data in notebooks.items():
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
    notebook = {
        "cells": notebook_data["cells"],
        "metadata": notebook_data["metadata"],
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Generated: {notebook_path}")

print("\\nAll notebooks generated successfully!")



