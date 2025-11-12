# Employee Dataset Analysis

## Comprehensive Data Science & Predictive Analytics Project

This project provides a complete analysis of an employee dataset, including exploratory data analysis, statistical analysis, univariate/bivariate/multivariate analysis, and machine learning predictions. The analysis is implemented in both Python and R.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Analysis Components](#analysis-components)
- [Results](#results)
- [License](#license)
- [Contributing](#contributing)

## 🎯 Project Overview

This project performs comprehensive data analysis on an employee dataset to gain insights into employee characteristics, salary patterns, and predictive modeling. The analysis includes:

1. **Exploratory Data Analysis (EDA)** - Data exploration, cleaning, and visualization
2. **Statistical Analysis** - Descriptive, inferential, and exploratory statistics
3. **Univariate, Bivariate, and Multivariate Analysis** - Analysis of individual and multiple variables
4. **Machine Learning Analysis** - Predictive modeling using various algorithms

## 📊 Dataset Description

### Dataset Information

- **Source**: Employee dataset (employees.csv)
- **Size**: ~1,001 records
- **Features**: 8 columns
- **License**: Please respect the original dataset's license (see [License](#license) section)

### Dataset Columns

- **First Name**: Employee's first name
- **Gender**: Employee's gender (Male, Female)
- **Start Date**: Employee's start date
- **Last Login Time**: Last login timestamp
- **Salary**: Employee's salary
- **Bonus %**: Employee's bonus percentage
- **Senior Management**: Boolean indicating senior management status
- **Team**: Employee's team/department

### Data Quality

- Some missing values in Gender, Team, and Senior Management fields
- Dates in various formats that require parsing
- Numerical variables (Salary, Bonus %) may contain outliers

## 📁 Project Structure

```
emplyees/
├── data/
│   ├── raw/                    # Raw dataset
│   │   └── employees.csv
│   └── processed/              # Processed/cleaned datasets
│       └── employees_cleaned.csv
├── notebooks/
│   ├── python/                 # Python Jupyter notebooks
│   │   ├── 01_EDA.ipynb
│   │   ├── 02_Statistical_Analysis.ipynb
│   │   ├── 03_Univariate_Bivariate_Multivariate.ipynb
│   │   └── 04_ML_Analysis.ipynb
│   └── r/                      # R Jupyter notebooks
│       ├── 01_EDA.ipynb
│       ├── 02_Statistical_Analysis.ipynb
│       ├── 03_Univariate_Bivariate_Multivariate.ipynb
│       └── 04_ML_Analysis.ipynb
├── scripts/
│   ├── python/                 # Python analysis scripts
│   │   ├── eda.py
│   │   ├── statistical_analysis.py
│   │   ├── univariate_bivariate_multivariate.py
│   │   └── ml_analysis.py
│   └── r/                      # R analysis scripts
│       ├── eda.R
│       ├── statistical_analysis.R
│       ├── univariate_bivariate_multivariate.R
│       ├── ml_analysis.R
│       └── install_packages.R
├── results/
│   ├── models/                 # Trained ML models
│   ├── plots/                  # Generated visualizations
│   └── tables/                 # Statistical tables and results
├── docs/                       # Documentation
├── .gitignore                  # Git ignore file
├── .gitattributes              # Git attributes file
├── CONTRIBUTING.md             # Contribution guidelines
├── LICENSE                     # License file
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment file
├── setup_project.sh            # Setup script (Linux/Mac)
├── setup_project.ps1           # Setup script (Windows)
└── README.md                   # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8+ or R 4.0+
- Jupyter Notebook or JupyterLab
- Git (for version control)

### Quick Setup

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   git clone <repository-url>
   cd emplyees
   ```

2. **Run the setup script** (optional, but recommended):
   ```bash
   # On Linux/Mac
   chmod +x setup_project.sh
   ./setup_project.sh
   
   # On Windows (PowerShell)
   .\setup_project.ps1
   ```

3. **Place your dataset**:
   - Copy `employees.csv` to `data/raw/employees.csv`

### Python Setup

1. **Create a virtual environment** (recommended):
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Or using conda
   conda env create -f environment.yml
   conda activate employees-analysis
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Jupyter** (if not already installed):
   ```bash
   pip install jupyter notebook
   ```

### R Setup

1. **Install R packages**:
   ```bash
   Rscript scripts/r/install_packages.R
   ```

   Or manually install required packages:
   ```r
   install.packages(c("tidyverse", "ggplot2", "caret", "randomForest", 
                     "xgboost", "corrplot", "VIM", "psych"))
   ```

2. **Install IRkernel for Jupyter** (optional, for R notebooks in Jupyter):
   ```r
   install.packages("IRkernel")
   IRkernel::installspec()
   ```

## 📖 Usage

### Running Python Analysis

#### Option 1: Using Jupyter Notebooks

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Navigate to the notebooks**:
   - Open `notebooks/python/01_EDA.ipynb` for exploratory data analysis
   - Open `notebooks/python/02_Statistical_Analysis.ipynb` for statistical analysis
   - Open `notebooks/python/03_Univariate_Bivariate_Multivariate.ipynb` for variable analysis
   - Open `notebooks/python/04_ML_Analysis.ipynb` for machine learning analysis

3. **Run all cells** to execute the analysis

#### Option 2: Using Python Scripts

Run the scripts directly from the command line:

```bash
# From the project root directory
cd scripts/python

# Run EDA
python eda.py

# Run statistical analysis
python statistical_analysis.py

# Run univariate/bivariate/multivariate analysis
python univariate_bivariate_multivariate.py

# Run ML analysis
python ml_analysis.py
```

### Running R Analysis

#### Option 1: Using R Notebooks

1. **Open RStudio** or use Jupyter with R kernel
2. **Open the R notebooks** in `notebooks/r/`
3. **Run all cells** to execute the analysis

#### Option 2: Using R Scripts

Run the scripts directly from R or RStudio:

```r
# Set working directory to project root
setwd("path/to/emplyees")

# Source the scripts
source("scripts/r/eda.R")
source("scripts/r/statistical_analysis.R")
source("scripts/r/univariate_bivariate_multivariate.R")
source("scripts/r/ml_analysis.R")
```

Or from the command line:

```bash
Rscript scripts/r/eda.R
Rscript scripts/r/statistical_analysis.R
Rscript scripts/r/univariate_bivariate_multivariate.R
Rscript scripts/r/ml_analysis.R
```

## 🔍 Analysis Components

### 1. Exploratory Data Analysis (EDA)

**Files**: `notebooks/python/01_EDA.ipynb`, `scripts/python/eda.py`, `notebooks/r/01_EDA.ipynb`, `scripts/r/eda.R`

**Components**:
- Data loading and overview
- Missing values analysis
- Data cleaning and preprocessing
- Numerical variable analysis (distributions, statistics)
- Categorical variable analysis
- Outlier detection
- Correlation analysis
- Relationship analysis
- Time series analysis (hiring trends)

**Outputs**:
- Cleaned dataset (`data/processed/employees_cleaned.csv`)
- Visualization plots (`results/plots/`)
- Summary statistics

### 2. Statistical Analysis

**Files**: `notebooks/python/02_Statistical_Analysis.ipynb`, `scripts/python/statistical_analysis.py`, `notebooks/r/02_Statistical_Analysis.ipynb`, `scripts/r/statistical_analysis.R`

**Components**:
- **Descriptive Statistics**: Mean, median, standard deviation, skewness, kurtosis
- **Inferential Statistics**:
  - T-tests (salary by gender, by senior management)
  - Chi-square tests (gender and senior management association)
  - ANOVA (salary across teams)
  - Mann-Whitney U tests (non-parametric tests)
- **Exploratory Statistical Analysis**:
  - Normality tests (Shapiro-Wilk, Kolmogorov-Smirnov)
  - Correlation analysis with significance testing
  - Confidence intervals

**Outputs**:
- Statistical tables (`results/tables/`)
- Hypothesis test results
- Visualization plots

### 3. Univariate, Bivariate, and Multivariate Analysis

**Files**: `notebooks/python/03_Univariate_Bivariate_Multivariate.ipynb`, `scripts/python/univariate_bivariate_multivariate.py`, `notebooks/r/03_Univariate_Bivariate_Multivariate.ipynb`, `scripts/r/univariate_bivariate_multivariate.R`

**Components**:
- **Univariate Analysis**:
  - Individual variable distributions
  - Histograms, box plots, Q-Q plots
  - Statistical summaries for each variable
- **Bivariate Analysis**:
  - Numerical vs Numerical (scatter plots, correlations)
  - Numerical vs Categorical (box plots, violin plots)
  - Categorical vs Categorical (contingency tables, chi-square tests)
- **Multivariate Analysis**:
  - Pairwise relationships (pair plots)
  - Multiple variable interactions
  - Heatmaps
  - 3D visualizations

**Outputs**:
- Analysis plots (`results/plots/`)
- Summary tables (`results/tables/`)

### 4. Machine Learning Analysis

**Files**: `notebooks/python/04_ML_Analysis.ipynb`, `scripts/python/ml_analysis.py`, `notebooks/r/04_ML_Analysis.ipynb`, `scripts/r/ml_analysis.R`

**Components**:
- **Data Preprocessing**:
  - Handling missing values
  - Feature engineering
  - Categorical encoding
  - Feature scaling
- **Model Training**:
  - Linear Regression
  - Ridge Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM (Python only)
- **Model Evaluation**:
  - R² score
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - Cross-validation
- **Feature Importance Analysis**
- **Prediction Visualization**

**Target Variables**:
- Salary prediction
- Bonus % prediction

**Outputs**:
- Trained models (`results/models/`)
- Model comparison tables (`results/tables/`)
- Feature importance plots (`results/plots/`)
- Prediction plots (`results/plots/`)

## 📈 Results

All results are saved in the `results/` directory:

- **Models**: Trained machine learning models (`.pkl` for Python, `.rds` for R)
- **Plots**: All visualization plots (`.png` files)
- **Tables**: Statistical tables and comparison results (`.csv` files)

### Key Findings

(Results will be generated when you run the analysis scripts)

## 📝 License

**Important**: Please respect the original dataset's license. The dataset used in this project (`employees.csv`) should be used in accordance with its original license terms. If you are using this dataset, please:

1. Check the original dataset's license terms
2. Attribute the dataset source appropriately
3. Comply with any restrictions specified in the license

This analysis code and project structure are provided for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines, coding standards, and best practices.

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

## 🙏 Acknowledgments

- Thanks to the creators of the employee dataset
- Open-source libraries and tools used in this project
- Data science community for inspiration and resources

## 📚 References

- Python Data Science Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
- R Statistical Computing: tidyverse, ggplot2, caret, randomForest
- Machine Learning: XGBoost, LightGBM, scikit-learn

---

**Note**: This project is for educational and research purposes. Always ensure you have the right to use and analyze the dataset according to its license terms.

