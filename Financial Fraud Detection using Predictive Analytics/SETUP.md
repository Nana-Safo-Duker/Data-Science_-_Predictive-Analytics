# Setup Guide

This guide will help you set up the project environment and run the analysis.

## Prerequisites

- Python 3.8 or higher
- R 4.0 or higher (optional, for R analysis)
- Git
- Jupyter Notebook or JupyterLab (for running notebooks)

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd fraud_data
```

### 2. Python Environment Setup

#### Create Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

#### Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### Verify Installation

```bash
python -c "import pandas, numpy, sklearn, xgboost, lightgbm; print('All packages installed successfully!')"
```

### 3. R Environment Setup (Optional)

#### Install R Packages

Open R or RStudio and run:

```r
# Install required packages
install.packages(c(
  "tidyverse",
  "data.table",
  "ggplot2",
  "caret",
  "randomForest",
  "xgboost",
  "corrplot",
  "VIM",
  "naniar",
  "pROC",
  "ROCR",
  "plotly",
  "psych",
  "car",
  "gridExtra",
  "dplyr"
))

# Install IRkernel for Jupyter
install.packages('IRkernel')
IRkernel::installspec()
```

#### Verify Installation

```r
library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
cat("All R packages installed successfully!\n")
```

### 4. Prepare Data

1. Place your fraud detection dataset in the `data/` directory
2. Ensure the dataset is named `fraud_data.csv`
3. Verify the dataset contains the required columns, including `isFraud` as the target variable

### 5. Create Output Directories

The output directories should already exist, but if not:

```bash
mkdir -p outputs/figures
mkdir -p outputs/models
mkdir -p reports/python
mkdir -p reports/r
```

### 6. Verify Setup

Run a quick test to verify everything is set up correctly:

**Python:**
```bash
python scripts/python/eda.py
```

**R:**
```bash
Rscript scripts/r/eda.R
```

## Running the Analysis

### Option 1: Using Jupyter Notebooks (Recommended)

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Navigate to the notebook you want to run:
   - `notebooks/python/01_EDA.ipynb` - Exploratory Data Analysis
   - `notebooks/python/02_Statistical_Analysis.ipynb` - Statistical Analysis
   - `notebooks/python/03_Univariate_Bivariate_Multivariate.ipynb` - Univariate/Bivariate/Multivariate Analysis
   - `notebooks/python/04_ML_Analysis.ipynb` - Machine Learning Analysis

3. Run all cells to execute the analysis

### Option 2: Using Python Scripts

```bash
# EDA
python scripts/python/eda.py

# Statistical Analysis
python scripts/python/statistical_analysis.py

# Univariate/Bivariate/Multivariate Analysis
python scripts/python/univariate_bivariate_multivariate.py

# Machine Learning Analysis
python scripts/python/ml_analysis.py
```

### Option 3: Using R Scripts

```bash
# EDA
Rscript scripts/r/eda.R

# Statistical Analysis
Rscript scripts/r/statistical_analysis.R

# Univariate/Bivariate/Multivariate Analysis
Rscript scripts/r/univariate_bivariate_multivariate.R

# Machine Learning Analysis
Rscript scripts/r/ml_analysis.R
```

## Troubleshooting

### Common Issues

1. **Module not found error (Python)**
   - Ensure virtual environment is activated
   - Reinstall requirements: `pip install -r requirements.txt`

2. **Package not found error (R)**
   - Install missing packages: `install.packages("package_name")`
   - Update package list: `update.packages()`

3. **File not found error**
   - Verify dataset is in `data/fraud_data.csv`
   - Check file paths in scripts/notebooks

4. **Memory errors**
   - Reduce dataset size for testing
   - Use sampling in analysis scripts
   - Increase system memory if possible

5. **Jupyter kernel not found**
   - Install IRkernel for R: `IRkernel::installspec()`
   - Select correct kernel in Jupyter notebook

## Next Steps

1. Review the README.md for project overview
2. Start with EDA to understand the data
3. Proceed through statistical analysis and ML modeling
4. Check outputs in `outputs/figures/` and `outputs/models/`

## Getting Help

- Check the README.md for detailed documentation
- Review notebook comments and docstrings
- Open an issue on GitHub for bugs or questions
- Check the CONTRIBUTING.md for contribution guidelines

## License

Please respect the original dataset's license. See LICENSE.md for details.

