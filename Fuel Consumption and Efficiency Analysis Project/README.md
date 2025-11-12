# Fuel Consumption Analysis Project

## Overview

This project provides a comprehensive analysis of fuel consumption and CO2 emissions data for various vehicles. The analysis includes exploratory data analysis (EDA), statistical analysis, univariate/bivariate/multivariate analysis, and machine learning models to predict fuel consumption and CO2 emissions.

## Project Structure

```
FuelConsumption/
│
├── data/
│   └── FuelConsumption.csv          # Original dataset
│
├── notebooks/
│   ├── python/
│   │   ├── 01_EDA.ipynb             # Exploratory Data Analysis (Python)
│   │   ├── 02_Statistical_Analysis.ipynb  # Statistical Analysis (Python)
│   │   ├── 03_Univariate_Bivariate_Multivariate_Analysis.ipynb  # Univariate/Bivariate/Multivariate Analysis (Python)
│   │   └── 04_ML_Analysis.ipynb     # Machine Learning Analysis (Python)
│   │
│   └── r/
│       ├── 01_EDA.ipynb             # Exploratory Data Analysis (R)
│       ├── 02_Statistical_Analysis.ipynb  # Statistical Analysis (R)
│       ├── 03_Univariate_Bivariate_Multivariate_Analysis.ipynb  # Univariate/Bivariate/Multivariate Analysis (R)
│       └── 04_ML_Analysis.ipynb     # Machine Learning Analysis (R)
│
├── scripts/
│   ├── python/
│   │   ├── eda.py                   # EDA script (Python)
│   │   ├── statistical_analysis.py  # Statistical analysis script (Python)
│   │   ├── univariate_bivariate_multivariate.py  # Analysis script (Python)
│   │   └── ml_analysis.py           # ML analysis script (Python)
│   │
│   └── r/
│       ├── eda.R                    # EDA script (R)
│       ├── statistical_analysis.R   # Statistical analysis script (R)
│       ├── univariate_bivariate_multivariate.R  # Analysis script (R)
│       └── ml_analysis.R            # ML analysis script (R)
│
├── outputs/
│   ├── figures/                     # Generated visualizations
│   └── models/                      # Trained ML models
│
├── docs/                            # Additional documentation
│
├── requirements.txt                 # Python dependencies
├── requirements_r.txt               # R package requirements
├── .gitignore                       # Git ignore file
└── README.md                        # This file

```

## Dataset Description

The dataset contains information about vehicle fuel consumption and CO2 emissions with the following features:

- **Year**: Model year
- **MAKE**: Vehicle manufacturer
- **MODEL**: Vehicle model
- **VEHICLE CLASS**: Vehicle classification
- **ENGINE SIZE**: Engine displacement in liters
- **CYLINDERS**: Number of cylinders
- **TRANSMISSION**: Transmission type
- **FUEL**: Fuel type (X = Regular gasoline, Z = Premium gasoline, etc.)
- **FUEL CONSUMPTION**: Fuel consumption in L/100km
- **COEMISSIONS**: CO2 emissions in g/km

## Features

### 1. Exploratory Data Analysis (EDA)
- Data overview and summary statistics
- Missing value analysis
- Distribution analysis
- Correlation analysis
- Categorical variable analysis
- Temporal trend analysis
- Outlier detection

### 2. Statistical Analysis
- **Descriptive Statistics**: Mean, median, mode, standard deviation, variance, skewness, kurtosis
- **Inferential Statistics**: Normality tests, t-tests, ANOVA
- **Exploratory Statistical Analysis**: Confidence intervals, correlation analysis with p-values

### 3. Univariate, Bivariate, and Multivariate Analysis
- **Univariate Analysis**: Individual variable distributions and statistics
- **Bivariate Analysis**: Relationships between pairs of variables
- **Multivariate Analysis**: Relationships among multiple variables using pair plots and correlation matrices

### 4. Machine Learning Analysis
- **Algorithms Used**:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
- **Target Variables**:
  - Fuel Consumption (L/100km)
  - CO2 Emissions (g/km)
- **Evaluation Metrics**: R² Score, RMSE, MAE, Cross-validation

## Installation

### Python Environment

1. Install Python 3.8 or higher
2. Install required packages:
```bash
pip install -r requirements.txt
```

### R Environment

1. Install R (version 4.0 or higher)
2. Install required packages:
```r
# Open R or RStudio and run:
install.packages(c("dplyr", "ggplot2", "corrplot", "VIM", "gridExtra", 
                   "readr", "caret", "randomForest", "e1071", "glmnet", 
                   "xgboost", "rpart", "rpart.plot", "rmarkdown", "knitr"))
```

## Usage

### Python

#### Running Notebooks
1. Start Jupyter Notebook:
```bash
jupyter notebook
```
2. Navigate to `notebooks/python/` and open the desired notebook

#### Running Scripts
```bash
# EDA
python scripts/python/eda.py

# Statistical Analysis
python scripts/python/statistical_analysis.py

# Univariate/Bivariate/Multivariate Analysis
python scripts/python/univariate_bivariate_multivariate.py

# ML Analysis
python scripts/python/ml_analysis.py
```

### R

#### Running Notebooks
1. Install IRkernel for Jupyter:
```r
install.packages('IRkernel')
IRkernel::installspec()
```
2. Start Jupyter Notebook and open R notebooks from `notebooks/r/`

#### Running Scripts
```r
# In R or RStudio
source("scripts/r/eda.R")
source("scripts/r/statistical_analysis.R")
source("scripts/r/univariate_bivariate_multivariate.R")
source("scripts/r/ml_analysis.R")
```

## Results

### Key Findings

1. **Strong Correlation**: Fuel consumption and CO2 emissions show a very strong positive correlation (r ≈ 0.98)
2. **Engine Characteristics**: Larger engine sizes and more cylinders are associated with higher fuel consumption and CO2 emissions
3. **Temporal Trends**: Analysis of trends over years reveals changes in vehicle efficiency
4. **Model Performance**: Random Forest and Gradient Boosting models achieve high R² scores (>0.95) for predicting both fuel consumption and CO2 emissions

### Output Files

- **Figures**: All visualizations are saved in `outputs/figures/`
- **Models**: Trained models are saved in `outputs/models/`

## License

This project respects the original dataset's license. Please refer to the dataset source for license information.

**Note**: This dataset is provided for educational and research purposes. Please ensure compliance with the original dataset's terms of use and licensing agreements.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Dataset source: [Please provide the original dataset source and citation]
- Thanks to all contributors and the open-source community

## Contact

For questions or issues, please open an issue on the repository.

---

**Last Updated**: 2024


