# Fraud Detection Data Science Project

## 📋 Overview

This project performs comprehensive data science analysis on fraud transaction data, including exploratory data analysis (EDA), statistical analysis, univariate/bivariate/multivariate analysis, and machine learning model development for fraud detection. The analysis is implemented in both **Python** and **R** to provide comprehensive insights and model comparisons.

## 🎯 Objectives

1. Perform comprehensive exploratory data analysis (EDA) to understand the dataset
2. Conduct descriptive, inferential, and exploratory statistical analysis
3. Perform univariate, bivariate, and multivariate analysis
4. Develop and evaluate machine learning models for fraud detection
5. Identify key features and patterns associated with fraudulent transactions

## 📁 Project Structure

```
fraud_data/
├── data/                              # Data files
│   ├── fraud_data.csv                 # Main dataset
│   └── README.md                      # Data documentation
├── notebooks/                         # Jupyter notebooks
│   ├── python/                       # Python notebooks
│   │   ├── 01_EDA.ipynb             # Exploratory Data Analysis
│   │   ├── 02_Statistical_Analysis.ipynb  # Statistical Analysis
│   │   ├── 03_Univariate_Bivariate_Multivariate.ipynb  # Univariate/Bivariate/Multivariate Analysis
│   │   └── 04_ML_Analysis.ipynb     # Machine Learning Analysis
│   └── r/                            # R notebooks
│       ├── 01_EDA.ipynb             # Exploratory Data Analysis (R)
│       ├── 02_Statistical_Analysis.ipynb  # Statistical Analysis (R)
│       ├── 03_Univariate_Bivariate_Multivariate.ipynb  # Univariate/Bivariate/Multivariate Analysis (R)
│       └── 04_ML_Analysis.ipynb     # Machine Learning Analysis (R)
├── scripts/                          # Python and R scripts
│   ├── python/
│   │   ├── eda.py                   # EDA script
│   │   ├── statistical_analysis.py  # Statistical analysis script
│   │   ├── univariate_bivariate_multivariate.py  # Univariate/Bivariate/Multivariate analysis script
│   │   └── ml_analysis.py           # Machine learning analysis script
│   └── r/
│       ├── eda.R                    # EDA script (R)
│       ├── statistical_analysis.R   # Statistical analysis script (R)
│       ├── univariate_bivariate_multivariate.R  # Univariate/Bivariate/Multivariate analysis script (R)
│       └── ml_analysis.R            # Machine learning analysis script (R)
├── outputs/                         # Generated outputs
│   ├── figures/                    # Visualizations
│   │   └── .gitkeep
│   └── models/                     # Trained models
│       └── .gitkeep
├── reports/                        # Analysis reports
│   ├── python/
│   └── r/
├── docs/                           # Documentation
├── requirements.txt                # Python dependencies
├── requirements_r.txt              # R package list
├── LICENSE.md                      # License information
├── .gitignore                      # Git ignore file
└── README.md                       # This file
```

## 📊 Dataset

### Description

The fraud dataset contains transaction information with various features including:

- **Transaction details**: Transaction ID, date, amount
- **Card information**: Card type, card numbers (card1-card6)
- **Address information**: Billing and shipping addresses (addr1, addr2)
- **Email domains**: Purchaser and recipient email domains
- **Device information**: Device type and browser information
- **Transaction features**: 
  - C features (C1-C14): Count features
  - D features (D1-D15): Time delta features
  - M features (M1-M9): Match features
  - V features (V1-V339): Vesta engineered features
- **Identity features**: id_01-id_38 (identity features)
- **Target variable**: `isFraud` (0 = legitimate transaction, 1 = fraudulent transaction)

### Dataset License

**⚠️ IMPORTANT: Please respect the original dataset's license.**

This project uses a fraud detection dataset. Before using this dataset, please ensure:

1. You have the appropriate permissions to use the dataset
2. You comply with the original dataset's license terms
3. You acknowledge the original data source appropriately

**License Compliance:**
- If you are using this dataset, you must comply with the original license terms
- This project code is provided for educational and research purposes
- The dataset itself is subject to its original license terms

**Usage:**
- You may use this project for educational purposes
- You may modify the code as needed
- You should respect data privacy and confidentiality requirements

For questions about dataset licensing, please contact the original dataset provider.

**Note:** This project assumes you have legal access to the fraud detection dataset. Ensure compliance with all applicable licenses and terms of use.

## 🚀 Installation

### Python Environment

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Jupyter (if not already installed):**
   ```bash
   pip install jupyter
   ```

### R Environment

1. **Install R packages:**
   ```r
   # Open R or RStudio
   install.packages(c("tidyverse", "data.table", "ggplot2", "caret", 
                      "randomForest", "xgboost", "corrplot", "VIM", 
                      "naniar", "pROC", "ROCR", "plotly", "psych", "car",
                      "gridExtra", "dplyr"))
   ```

2. **Install IRkernel for Jupyter:**
   ```r
   install.packages('IRkernel')
   IRkernel::installspec()
   ```

## 📖 Usage

### Running Python Analysis

#### 1. Exploratory Data Analysis (EDA)

**Using Jupyter Notebook:**
```bash
jupyter notebook notebooks/python/01_EDA.ipynb
```

**Using Python script:**
```bash
python scripts/python/eda.py
```

#### 2. Statistical Analysis

**Using Jupyter Notebook:**
```bash
jupyter notebook notebooks/python/02_Statistical_Analysis.ipynb
```

**Using Python script:**
```bash
python scripts/python/statistical_analysis.py
```

#### 3. Univariate, Bivariate, and Multivariate Analysis

**Using Jupyter Notebook:**
```bash
jupyter notebook notebooks/python/03_Univariate_Bivariate_Multivariate.ipynb
```

**Using Python script:**
```bash
python scripts/python/univariate_bivariate_multivariate.py
```

#### 4. Machine Learning Analysis

**Using Jupyter Notebook:**
```bash
jupyter notebook notebooks/python/04_ML_Analysis.ipynb
```

**Using Python script:**
```bash
python scripts/python/ml_analysis.py
```

### Running R Analysis

#### 1. Exploratory Data Analysis (EDA)

**Using Jupyter Notebook:**
```bash
jupyter notebook notebooks/r/01_EDA.ipynb
```

**Using R script:**
```bash
Rscript scripts/r/eda.R
```

#### 2. Statistical Analysis

**Using Jupyter Notebook:**
```bash
jupyter notebook notebooks/r/02_Statistical_Analysis.ipynb
```

**Using R script:**
```bash
Rscript scripts/r/statistical_analysis.R
```

#### 3. Univariate, Bivariate, and Multivariate Analysis

**Using Jupyter Notebook:**
```bash
jupyter notebook notebooks/r/03_Univariate_Bivariate_Multivariate.ipynb
```

**Using R script:**
```bash
Rscript scripts/r/univariate_bivariate_multivariate.R
```

#### 4. Machine Learning Analysis

**Using Jupyter Notebook:**
```bash
jupyter notebook notebooks/r/04_ML_Analysis.ipynb
```

**Using R script:**
```bash
Rscript scripts/r/ml_analysis.R
```

## 📈 Analysis Components

### 1. Exploratory Data Analysis (EDA)

- **Data loading and inspection**: Load dataset and examine structure
- **Missing value analysis**: Identify and analyze missing values
- **Data types and summary statistics**: Understand data types and basic statistics
- **Distribution analysis**: Analyze distributions of key features
- **Target variable analysis**: Examine fraud distribution and class imbalance
- **Feature engineering insights**: Identify potential features for modeling
- **Time-based analysis**: Analyze temporal patterns in transactions
- **Correlation analysis**: Identify correlations between features

### 2. Statistical Analysis

#### Descriptive Statistics
- **Central tendencies**: Mean, median, mode
- **Dispersion measures**: Standard deviation, variance, range, IQR
- **Shape measures**: Skewness, kurtosis
- **Summary statistics by fraud status**: Compare statistics between fraud and legitimate transactions

#### Inferential Statistics
- **Hypothesis testing**: Test differences between fraud and legitimate transactions
- **Non-parametric tests**: Mann-Whitney U test (for non-normal distributions)
- **Parametric tests**: Independent t-test (for comparison)
- **Chi-square tests**: Test associations between categorical variables and fraud
- **Confidence intervals**: Calculate 95% confidence intervals for key metrics

#### Exploratory Statistics
- **Correlation analysis**: Analyze correlations between features and fraud
- **Statistical significance**: Identify statistically significant relationships
- **Feature relationships**: Explore relationships between features and fraud status

### 3. Univariate, Bivariate, and Multivariate Analysis

#### Univariate Analysis
- **Individual variable distributions**: Histograms, box plots, density plots
- **Statistical measures**: Mean, median, mode, std, variance, skewness, kurtosis
- **Normality tests**: Test for normal distribution
- **Categorical variable analysis**: Frequency distributions and mode analysis

#### Bivariate Analysis
- **Numerical vs Numerical**: Correlation analysis, scatter plots, pair plots
- **Numerical vs Categorical**: Box plots, violin plots, statistical tests
- **Categorical vs Categorical**: Contingency tables, chi-square tests
- **Target variable relationships**: Analyze relationships with fraud status

#### Multivariate Analysis
- **Correlation matrices**: Analyze correlations among multiple variables
- **Principal Component Analysis (PCA)**: Dimensionality reduction and feature extraction
- **Cluster Analysis (K-Means)**: Identify patterns and clusters in data
- **Feature importance**: Identify features most correlated with fraud

### 4. Machine Learning Analysis

#### Data Preprocessing
- **Feature selection**: Select key features for modeling
- **Missing value handling**: Handle missing values appropriately
- **Feature scaling**: Standardize features for certain algorithms
- **Train-test split**: Split data into training and testing sets
- **Class imbalance handling**: Address class imbalance using techniques like SMOTE or class weights

#### Models Implemented

**Python:**
- **Logistic Regression**: Baseline linear model
- **Random Forest**: Ensemble tree-based model
- **XGBoost**: Gradient boosting model
- **LightGBM**: Lightweight gradient boosting model

**R:**
- **Logistic Regression**: Baseline linear model
- **Random Forest**: Ensemble tree-based model
- **XGBoost**: Gradient boosting model

#### Model Evaluation
- **Classification metrics**: Accuracy, precision, recall, F1-score
- **ROC curves**: Receiver Operating Characteristic curves
- **AUC-ROC**: Area Under the ROC Curve
- **Confusion matrices**: True positives, false positives, true negatives, false negatives
- **Feature importance**: Identify most important features for prediction

#### Model Comparison
- **Performance comparison**: Compare models based on AUC-ROC and other metrics
- **Best model selection**: Select the best performing model
- **Model interpretation**: Interpret model results and feature importance

## 🔍 Key Findings

### Data Characteristics
- **Class imbalance**: Fraudulent transactions are typically a small percentage of all transactions
- **Feature diversity**: Dataset contains various types of features (numerical, categorical, engineered)
- **Missing values**: Some features may contain missing values that need to be handled

### Statistical Insights
- **Transaction amount differences**: Fraudulent transactions may have different amount distributions
- **Feature correlations**: Certain features are more correlated with fraud than others
- **Temporal patterns**: Fraud may occur more frequently at certain times

### Model Performance
- **Best algorithms**: Gradient boosting algorithms (XGBoost, LightGBM) typically perform best
- **Feature importance**: Transaction amount and card-related features are often most important
- **Evaluation metrics**: AUC-ROC is a key metric for fraud detection due to class imbalance

## 📊 Results

Results and visualizations are saved in the `outputs/` directory:

- **`outputs/figures/`**: All generated plots and charts
  - EDA visualizations
  - Statistical analysis plots
  - Univariate/bivariate/multivariate analysis plots
  - Machine learning evaluation plots (ROC curves, feature importance, etc.)

- **`outputs/models/`**: Trained model files
  - Best model (typically XGBoost or LightGBM)
  - Scaler/preprocessor (if used)

- **`reports/`**: Analysis reports (if generated)

## 🛠️ Technologies Used

### Python
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **Statistical Analysis**: scipy, statsmodels
- **Imbalanced Learning**: imbalanced-learn
- **Model Interpretation**: shap

### R
- **Data Processing**: tidyverse, data.table, dplyr
- **Visualization**: ggplot2, plotly, corrplot
- **Machine Learning**: caret, randomForest, xgboost
- **Statistical Analysis**: psych, car
- **Model Evaluation**: pROC, ROCR

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

**Important:** Please respect the original dataset's license. This project code is provided as-is for educational and research purposes.

- The project code is available for educational and research use
- The dataset itself is subject to its original license terms
- Please ensure you have legal access to the dataset before using this project

## 🙏 Acknowledgments

- Dataset providers (acknowledge original source)
- Open-source libraries and tools used in this project:
  - Python: pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, seaborn
  - R: tidyverse, caret, randomForest, xgboost, ggplot2
- Contributors and reviewers

## 📧 Contact

For questions or issues, please open an issue on the GitHub repository.

## 📚 References

- Fraud Detection Best Practices
- Machine Learning for Fraud Detection
- Statistical Analysis for Financial Data
- Imbalanced Learning Techniques

## 📅 Version History

- **v1.0.0** - Initial release with comprehensive EDA, statistical analysis, and ML models
  - Complete EDA implementation in Python and R
  - Comprehensive statistical analysis
  - Univariate, bivariate, and multivariate analysis
  - Machine learning models (Logistic Regression, Random Forest, XGBoost, LightGBM)
  - Feature importance analysis
  - Model evaluation and comparison

## 🎓 Educational Purpose

This project is designed for educational purposes to demonstrate:
- Comprehensive data science workflow
- Statistical analysis techniques
- Machine learning model development
- Fraud detection methodologies
- Best practices in data science projects

---

**Note:** This project assumes you have legal access to the fraud detection dataset. Ensure compliance with all applicable licenses and terms of use before proceeding.
