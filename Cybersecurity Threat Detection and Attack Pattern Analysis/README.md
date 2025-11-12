# Cybersecurity Attacks Analysis

A comprehensive data science project analyzing cybersecurity attack patterns using exploratory data analysis, statistical analysis, and machine learning techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Analysis Components](#analysis-components)
- [Results](#results)
- [License](#license)
- [Contributing](#contributing)

## 🎯 Overview

This project provides a comprehensive analysis of cybersecurity attacks dataset, including:

- **Exploratory Data Analysis (EDA)**: Understanding dataset structure, distributions, and patterns
- **Statistical Analysis**: Descriptive, inferential, and exploratory statistical analysis
- **Univariate, Bivariate, and Multivariate Analysis**: Analysis of individual variables and relationships
- **Machine Learning Analysis**: Classification models to predict attack categories

The analysis is performed using both **Python** and **R** programming languages.

## 📊 Dataset

### Dataset Information

- **Name**: Cybersecurity Attacks Dataset
- **Size**: 178,031 records
- **Features**: 11 columns
- **Description**: Contains information about various cybersecurity attacks including attack categories, protocols, IP addresses, ports, and timestamps

### Dataset Columns

- `Attack category`: Type of attack (Reconnaissance, Exploits, DoS, etc.)
- `Attack subcategory`: Subcategory of the attack
- `Protocol`: Network protocol used (TCP, UDP, etc.)
- `Source IP`: Source IP address
- `Source Port`: Source port number
- `Destination IP`: Destination IP address
- `Destination Port`: Destination port number
- `Attack Name`: Name of the attack
- `Attack Reference`: Reference information (CVE, BID, etc.)
- `Time`: Timestamp of the attack (Unix timestamp)

### Dataset License

**Important**: Please respect the original dataset's license. This dataset is used for educational and research purposes. If you plan to use this dataset, ensure you have the proper rights and comply with the original license terms.

The dataset may contain sensitive information (IP addresses). Use responsibly and in accordance with privacy regulations.

## 📁 Project Structure

```
Cybersecurity_attacks/
│
├── data/
│   └── Cybersecurity_attacks.csv          # Original dataset
│
├── notebooks/
│   ├── python/
│   │   ├── 01_EDA_Cybersecurity_Attacks.ipynb
│   │   ├── 02_Statistical_Analysis.ipynb
│   │   ├── 03_Univariate_Bivariate_Multivariate_Analysis.ipynb
│   │   └── 04_ML_Analysis.ipynb
│   │
│   └── r/
│       ├── 01_EDA_Cybersecurity_Attacks.ipynb
│       ├── 02_Statistical_Analysis.ipynb
│       ├── 03_Univariate_Bivariate_Multivariate_Analysis.ipynb
│       └── 04_ML_Analysis.ipynb
│
├── scripts/
│   ├── python/
│   │   ├── eda.py                         # EDA script
│   │   ├── ml_analysis.py                 # ML analysis script
│   │   └── create_notebooks.py            # Notebook generation script
│   │
│   └── r/
│       └── install.R                      # R package installation script
│
├── visualizations/                        # Generated visualizations
├── results/                               # Analysis results and outputs
├── docs/                                  # Documentation
│
├── requirements.txt                       # Python dependencies
├── .gitignore                            # Git ignore file
├── LICENSE                               # License file
└── README.md                             # This file
```

## 🔧 Requirements

### Python Requirements

- Python 3.8 or higher
- Required packages listed in `requirements.txt`

### R Requirements

- R 4.0 or higher
- Required packages listed in `scripts/r/install.R`

## 🚀 Installation

### Python Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Cybersecurity_attacks
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### R Environment Setup

1. Install R packages:
```r
source("scripts/r/install.R")
```

2. For Jupyter notebook support with R:
```r
install.packages("IRkernel")
IRkernel::installspec()
```

## 📖 Usage

### Python Analysis

#### Run EDA Script
```bash
python scripts/python/eda.py
```

#### Run ML Analysis Script
```bash
python scripts/python/ml_analysis.py
```

#### Run Jupyter Notebooks
```bash
jupyter notebook notebooks/python/
```

### R Analysis

#### Run R Scripts
```r
source("scripts/r/install.R")
```

#### Run Jupyter Notebooks with R Kernel
```bash
jupyter notebook notebooks/r/
```

## 📈 Analysis Components

### 1. Exploratory Data Analysis (EDA)

**Python**: `notebooks/python/01_EDA_Cybersecurity_Attacks.ipynb`
**R**: `notebooks/r/01_EDA_Cybersecurity_Attacks.ipynb`

- Data loading and inspection
- Missing values analysis
- Data cleaning and preprocessing
- Categorical variables analysis
- Numerical variables analysis
- Temporal analysis
- IP address analysis
- Summary statistics and insights

### 2. Statistical Analysis

**Python**: `notebooks/python/02_Statistical_Analysis.ipynb`
**R**: `notebooks/r/02_Statistical_Analysis.ipynb`

- Descriptive statistics
- Normality tests
- Hypothesis testing:
  - Chi-square test for independence
  - ANOVA (Analysis of Variance)
  - Mann-Whitney U test
- Confidence intervals
- Correlation analysis
- Statistical summary report

### 3. Univariate, Bivariate, and Multivariate Analysis

**Python**: `notebooks/python/03_Univariate_Bivariate_Multivariate_Analysis.ipynb`
**R**: `notebooks/r/03_Univariate_Bivariate_Multivariate_Analysis.ipynb`

#### Univariate Analysis
- Numerical variables: histograms, box plots, Q-Q plots
- Categorical variables: frequency distributions, bar charts, pie charts

#### Bivariate Analysis
- Numerical vs Numerical: scatter plots, correlation analysis
- Categorical vs Numerical: box plots, violin plots
- Categorical vs Categorical: contingency tables, heatmaps

#### Multivariate Analysis
- Correlation matrices
- Pair plots
- Multivariate visualizations

### 4. Machine Learning Analysis

**Python**: `notebooks/python/04_ML_Analysis.ipynb`
**R**: `notebooks/r/04_ML_Analysis.ipynb`

#### Models Implemented

1. **Random Forest Classifier**
2. **Gradient Boosting Classifier**
3. **XGBoost Classifier**
4. **LightGBM Classifier**
5. **Logistic Regression**
6. **Support Vector Machine (SVM)**
7. **Naive Bayes**

#### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report

#### Feature Engineering

- Time-based features (hour, day of week, month)
- Protocol encoding
- Port analysis
- Temporal patterns

## 📊 Results

### Key Findings

1. **Attack Categories Distribution**: The dataset contains various attack types with different frequencies
2. **Protocol Analysis**: TCP and UDP are the most common protocols
3. **Temporal Patterns**: Certain hours and days show higher attack frequencies
4. **Port Analysis**: Common destination ports are identified
5. **IP Address Patterns**: Top source and destination IPs are identified

### Model Performance

The machine learning models are evaluated based on multiple metrics. The best performing model is selected based on F1-score, considering the imbalanced nature of the dataset.

### Visualizations

All visualizations are saved in the `visualizations/` directory, including:
- Distribution plots
- Correlation matrices
- Temporal analysis plots
- Model performance metrics
- Feature importance plots

## 🔒 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Dataset License Note**: The original dataset (Cybersecurity_attacks.csv) is used for educational and research purposes. Please ensure you have the proper rights to use this dataset and comply with its original license terms.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Contribution Guidelines

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Notes

- This project is for educational and research purposes
- The dataset may contain sensitive information - use responsibly
- Ensure compliance with privacy regulations when handling IP addresses
- Respect the original dataset's license terms

## 🙏 Acknowledgments

- Dataset source: [Please add dataset source/acknowledgment]
- Libraries and tools used: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, lightgbm, and others

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Last Updated**: 2024

**Version**: 1.0.0



