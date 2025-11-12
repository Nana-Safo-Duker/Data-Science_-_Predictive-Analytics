# Quick Start Guide

## 🚀 Getting Started Quickly

### 1. Setup Environment

#### Python Setup
```bash
# Install Python dependencies
pip install -r requirements.txt
```

#### R Setup
```r
# Install R packages
install.packages(c("tidyverse", "caret", "tm", "ggplot2", "e1071", 
                   "randomForest", "xgboost", "naivebayes", "corrplot", 
                   "psych", "Hmisc", "wordcloud", "RColorBrewer"))
```

### 2. Run Analysis

#### Option A: Run Python Scripts
```bash
# Navigate to scripts directory
cd scripts/python

# Run EDA
python eda.py

# Run Statistical Analysis
python statistical_analysis.py

# Run Univariate, Bivariate, Multivariate Analysis
python univariate_bivariate_multivariate.py

# Run Machine Learning Analysis
python ml_analysis.py
```

#### Option B: Run R Scripts
```r
# Set working directory
setwd("scripts/R")

# Run EDA
source("eda.R")

# Run Statistical Analysis
source("statistical_analysis.R")

# Run Univariate, Bivariate, Multivariate Analysis
source("univariate_bivariate_multivariate.R")

# Run Machine Learning Analysis
source("ml_analysis.R")
```

#### Option C: Use Jupyter Notebooks
```bash
# Start Jupyter Notebook
jupyter notebook

# Open and run notebooks from the notebooks/ directory
# - notebooks/python/01_EDA_Python.ipynb
# - notebooks/python/02_Statistical_Analysis_Python.ipynb
# - notebooks/python/03_Univariate_Bivariate_Multivariate_Analysis.ipynb
# - notebooks/python/04_ML_Analysis_Python.ipynb
# - notebooks/R/01_EDA_R.ipynb
# - notebooks/R/02_Statistical_Analysis_R.ipynb
# - notebooks/R/03_Univariate_Bivariate_Multivariate_Analysis_R.ipynb
# - notebooks/R/04_ML_Analysis_R.ipynb
```

### 3. View Results

- **Figures**: Check `output/figures/` directory for generated visualizations
- **Models**: Check `models/` directory for saved machine learning models
- **Processed Data**: Check `data/` directory for processed datasets

### 4. Project Structure

```
emails-spam/
├── data/                          # Datasets
├── notebooks/                     # Jupyter notebooks
│   ├── python/                   # Python notebooks
│   └── R/                        # R notebooks
├── scripts/                      # Analysis scripts
│   ├── python/                   # Python scripts
│   └── R/                        # R scripts
├── models/                       # Saved models
├── output/                       # Output files
│   └── figures/                 # Generated figures
└── requirements.txt              # Python dependencies
```

### 5. Key Features

- ✅ Comprehensive EDA
- ✅ Statistical Analysis (Descriptive, Inferential, Exploratory)
- ✅ Univariate, Bivariate, Multivariate Analysis
- ✅ Machine Learning Models (Naive Bayes, SVM, Random Forest, XGBoost, Logistic Regression)
- ✅ Model Comparison and Evaluation
- ✅ Visualizations and Reports

### 6. Next Steps

1. Run the analysis scripts or notebooks
2. Review the generated visualizations in `output/figures/`
3. Check model performance metrics
4. Explore the results and insights
5. Customize the analysis as needed

## 📚 Documentation

For more detailed information, see:
- [README.md](README.md) - Complete project documentation
- [DATASET_LICENSE.md](DATASET_LICENSE.md) - Dataset license information
- [LICENSE](LICENSE) - Project license

## 🆘 Troubleshooting

### Common Issues

1. **Missing dependencies**: Install required packages using `pip install -r requirements.txt` or R package installer
2. **Path issues**: Make sure you're running scripts from the correct directory
3. **Data not found**: Ensure the dataset is in the `data/` directory
4. **Memory issues**: For large datasets, consider sampling or increasing memory

### Getting Help

- Check the README.md for detailed documentation
- Review the script comments for guidance
- Open an issue in the repository if you encounter problems

---

**Happy Analyzing! 🎉**


