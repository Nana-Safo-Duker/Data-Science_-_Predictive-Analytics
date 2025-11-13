# Position Salaries - Data Science & Predictive Analytics Project

## Overview

This project provides a comprehensive analysis of position salaries using data science and predictive analytics techniques. The project includes exploratory data analysis, statistical analysis, and machine learning models to predict salaries based on position levels.

## Project Structure

```
Position_Salaries/
│
├── data/
│   ├── raw/                          # Raw dataset
│   │   └── Position_Salaries.csv
│   └── processed/                    # Processed datasets
│       ├── processed_data.csv
│       └── analysis_data.csv
│
├── notebooks/
│   ├── python/                       # Python notebooks
│   │   ├── 01_EDA.ipynb             # Exploratory Data Analysis
│   │   ├── 02_Statistical_Analysis.ipynb  # Statistical Analysis
│   │   ├── 03_Univariate_Bivariate_Multivariate_Analysis.ipynb
│   │   └── 04_ML_Analysis.ipynb     # Machine Learning Analysis
│   │
│   └── r/                            # R notebooks
│       ├── 01_EDA.Rmd               # Exploratory Data Analysis
│       ├── 02_Statistical_Analysis.Rmd  # Statistical Analysis
│       └── 03_ML_Analysis.Rmd       # Machine Learning Analysis
│
├── scripts/
│   ├── python/                       # Python scripts
│   │   └── 01_EDA.py
│   │
│   └── r/                            # R scripts
│       └── 01_EDA.R
│
├── results/
│   ├── figures/                      # Generated visualizations
│   └── models/                       # Saved ML models
│
├── requirements.txt                  # Python dependencies
├── environment.yml                   # Conda environment
├── .gitignore
└── README.md                         # This file
```

## Dataset

### Description
The Position Salaries dataset contains information about different job positions, their levels, and corresponding salaries. This dataset is commonly used for regression analysis and salary prediction tasks.

### Dataset Structure
- **Position**: Job position title
- **Level**: Position level (1-10)
- **Salary**: Annual salary in USD

### Dataset License
This dataset is provided for educational and research purposes. The original dataset is commonly used in machine learning courses and tutorials. 

**Note**: This is a synthetic/example dataset typically used for educational purposes. If you are using this dataset, please ensure you comply with any applicable data usage policies and cite the original source if required.

For educational datasets, common licenses include:
- **CC0 1.0 Universal (CC0 1.0) Public Domain Dedication** - Allows free use for any purpose
- **MIT License** - Permits reuse with attribution
- **Apache License 2.0** - Allows modification and distribution

**Recommendation**: If using this dataset in a production environment or for commercial purposes, verify the original source and ensure proper licensing compliance.

## Installation

### Python Environment

1. **Using pip:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Using conda:**
   ```bash
   conda env create -f environment.yml
   conda activate position_salaries
   ```

### R Environment

Install required R packages:
```r
install.packages(c("tidyverse", "ggplot2", "dplyr", "readr", "corrplot", "moments", 
                   "randomForest", "e1071", "caret"))
```

## Usage

### Python Analysis

1. **Exploratory Data Analysis:**
   ```bash
   # Run Jupyter notebook
   jupyter notebook notebooks/python/01_EDA.ipynb
   
   # Or run script
   python scripts/python/01_EDA.py
   ```

2. **Statistical Analysis:**
   ```bash
   jupyter notebook notebooks/python/02_Statistical_Analysis.ipynb
   ```

3. **Univariate/Bivariate/Multivariate Analysis:**
   ```bash
   jupyter notebook notebooks/python/03_Univariate_Bivariate_Multivariate_Analysis.ipynb
   ```

4. **Machine Learning Analysis:**
   ```bash
   jupyter notebook notebooks/python/04_ML_Analysis.ipynb
   ```

### R Analysis

1. **Exploratory Data Analysis:**
   ```r
   # Render R Markdown
   rmarkdown::render("notebooks/r/01_EDA.Rmd")
   
   # Or run script
   Rscript scripts/r/01_EDA.R
   ```

2. **Statistical Analysis:**
   ```r
   rmarkdown::render("notebooks/r/02_Statistical_Analysis.Rmd")
   ```

## Analysis Components

### 1. Exploratory Data Analysis (EDA)
- Data loading and inspection
- Missing value analysis
- Statistical summaries
- Distribution analysis
- Correlation analysis
- Data visualizations

### 2. Statistical Analysis
- **Descriptive Statistics**: Mean, median, mode, standard deviation, quartiles, skewness, kurtosis
- **Inferential Statistics**: 
  - Normality tests (Shapiro-Wilk, D'Agostino)
  - Correlation tests (Pearson, Spearman)
  - One-sample t-tests
  - Confidence intervals
- **Exploratory Statistical Analysis**: Q-Q plots, residual analysis

### 3. Univariate, Bivariate, and Multivariate Analysis
- **Univariate Analysis**: Individual variable analysis (Salary, Level)
- **Bivariate Analysis**: Relationship between Level and Salary
- **Multivariate Analysis**: Multiple variable analysis with feature engineering

### 4. Machine Learning Analysis
- **Linear Regression**: Basic linear model
- **Polynomial Regression**: Polynomial models of various degrees (2, 3, 4)
- **Random Forest Regression**: Ensemble learning approach
- **Support Vector Regression (SVR)**: Non-linear regression with kernels
- **Model Comparison**: Performance metrics (MSE, RMSE, MAE, R²)
- **Model Selection**: Best model selection based on performance
- **Predictions**: Salary predictions for new position levels

## Key Findings

### Data Insights
- Strong positive correlation between position level and salary
- Non-linear relationship between level and salary
- Salary increases exponentially with position level

### Model Performance
- **Polynomial Regression (Degree 4)** typically performs best for this dataset
- High R² score indicating good model fit
- Polynomial regression captures the non-linear relationship effectively

## Results

All results are saved in the `results/` directory:
- **Figures**: All visualizations saved as high-resolution PNG files
- **Models**: Trained ML models saved as pickle files (Python) or RDS files (R)
- **Predictions**: Predicted salaries for new position levels

## Dependencies

### Python
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scipy >= 1.10.0
- scikit-learn >= 1.3.0
- jupyter >= 1.0.0
- statsmodels >= 0.14.0
- plotly >= 5.14.0

### R
- tidyverse
- ggplot2
- dplyr
- readr
- corrplot
- moments
- randomForest
- e1071
- caret

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is provided for educational and research purposes. Please ensure compliance with dataset licensing requirements when using this project.

## Acknowledgments

- Dataset used for educational purposes
- Inspired by common machine learning regression problems
- Built with Python and R for comprehensive analysis

## Contact

For questions or suggestions, please open an issue in the repository.

## References

1. Scikit-learn Documentation: https://scikit-learn.org/
2. R Documentation: https://www.r-project.org/
3. Pandas Documentation: https://pandas.pydata.org/
4. Tidyverse Documentation: https://www.tidyverse.org/

---

**Note**: This project is designed for educational purposes. For production use, ensure proper data validation, model evaluation, and compliance with data usage policies.

---
*Enhanced with salary prediction modeling using regression techniques and comprehensive feature analysis*


