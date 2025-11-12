# Analysis Guide

## Overview

This guide provides instructions for running the analyses in this project.

## Prerequisites

### Python Setup

1. Install Python 3.8 or higher
2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### R Setup

1. Install R 4.0 or higher
2. Install RStudio (optional but recommended)
3. Install required packages:
```r
source("scripts/r/install_packages.R")
```

## Running Analyses

### Python Analyses

#### Option 1: Jupyter Notebooks

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Navigate to `notebooks/python/` and open notebooks in order

#### Option 2: Python Scripts

```bash
# Run EDA
python scripts/python/eda.py

# Run analysis
python scripts/python/univariate_bivariate_multivariate_analysis.py
```

### R Analyses

#### Option 1: RStudio

1. Open RStudio
2. Open `.Rmd` files from `notebooks/r/` directory
3. Click "Knit" to render HTML reports

#### Option 2: R Console

```r
# Render R Markdown files
rmarkdown::render("notebooks/r/01_EDA_Unicorn_Companies.Rmd")
rmarkdown::render("notebooks/r/02_Statistical_Analysis.Rmd")
rmarkdown::render("notebooks/r/03_Univariate_Bivariate_Multivariate_Analysis.Rmd")
rmarkdown::render("notebooks/r/04_ML_Analysis.Rmd")
```

## Analysis Steps

### Step 1: Exploratory Data Analysis

**Purpose**: Understand and clean the data

**Outputs**:
- Cleaned dataset (`data/Unicorn_Companies_cleaned.csv`)
- Initial visualizations
- Missing value analysis

**Key Steps**:
1. Load dataset
2. Inspect data structure
3. Identify missing values
4. Clean and preprocess data
5. Create initial visualizations
6. Save cleaned dataset

### Step 2: Statistical Analysis

**Purpose**: Perform descriptive and inferential statistics

**Outputs**:
- Statistical summaries
- Hypothesis test results
- Correlation matrices
- Confidence intervals

**Key Steps**:
1. Load cleaned dataset
2. Calculate descriptive statistics
3. Perform hypothesis tests
4. Analyze correlations
5. Calculate confidence intervals

### Step 3: Univariate, Bivariate, and Multivariate Analysis

**Purpose**: Analyze relationships between variables

**Outputs**:
- Distribution plots
- Relationship visualizations
- Correlation heatmaps
- Multivariate plots

**Key Steps**:
1. Load cleaned dataset
2. Perform univariate analysis
3. Perform bivariate analysis
4. Perform multivariate analysis
5. Generate comprehensive visualizations

### Step 4: Machine Learning Analysis

**Purpose**: Build predictive models

**Outputs**:
- Trained models
- Model evaluation metrics
- Feature importance plots
- Prediction visualizations

**Key Steps**:
1. Load cleaned dataset
2. Feature engineering
3. Split data into train/test sets
4. Train models
5. Evaluate models
6. Analyze feature importance
7. Save models

## Interpreting Results

### EDA Results

- **Missing Values**: Identify columns with missing data
- **Distributions**: Understand data distributions
- **Outliers**: Identify potential outliers

### Statistical Analysis Results

- **Descriptive Statistics**: Understand central tendencies and variability
- **Hypothesis Tests**: Determine if differences are statistically significant
- **Correlations**: Identify relationships between variables

### Analysis Results

- **Univariate**: Understand individual variable distributions
- **Bivariate**: Understand relationships between pairs of variables
- **Multivariate**: Understand complex relationships

### ML Results

- **Regression Metrics**: MSE, RMSE, R² score
- **Classification Metrics**: Accuracy, Precision, Recall, F1-score
- **Feature Importance**: Identify most important features

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install required packages
2. **File Not Found**: Check file paths
3. **Memory Issues**: Process data in chunks
4. **Convergence Issues**: Adjust model parameters

### Getting Help

1. Check error messages
2. Review documentation
3. Search for similar issues
4. Open an issue on GitHub

## Best Practices

1. **Run analyses in order**: EDA → Statistics → Analysis → ML
2. **Save intermediate results**: Don't rerun everything
3. **Use consistent random seeds**: For reproducibility
4. **Document findings**: Add comments and markdown cells
5. **Version control**: Commit code regularly

## Next Steps

After completing the analyses:

1. Review results and insights
2. Create summary reports
3. Share findings with stakeholders
4. Consider additional analyses
5. Deploy models (if applicable)


