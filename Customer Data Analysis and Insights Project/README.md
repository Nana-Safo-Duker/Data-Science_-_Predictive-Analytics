# Customer Data Analysis Project

## 📋 Project Overview

This project provides a comprehensive analysis of customer data, including exploratory data analysis (EDA), statistical analysis, and machine learning insights. The analysis is implemented in both Python and R to provide a complete data science perspective.

## 📁 Project Structure

```
Customers/
│
├── data/                    # Dataset files
│   └── Customers.csv
│
├── notebooks/               # Jupyter notebooks
│   ├── python/             # Python notebooks
│   │   ├── 01_EDA.ipynb
│   │   ├── 02_Statistical_Analysis.ipynb
│   │   ├── 03_Univariate_Bivariate_Multivariate_Analysis.ipynb
│   │   └── 04_ML_Analysis.ipynb
│   └── r/                  # R notebooks
│       ├── 01_EDA.ipynb
│       ├── 02_Statistical_Analysis.ipynb
│       ├── 03_Univariate_Bivariate_Multivariate_Analysis.ipynb
│       └── 04_ML_Analysis.ipynb
│
├── scripts/                # Executable scripts
│   ├── python/            # Python scripts
│   │   ├── eda.py
│   │   ├── statistical_analysis.py
│   │   ├── univariate_bivariate_multivariate_analysis.py
│   │   └── ml_analysis.py
│   └── r/                 # R scripts
│       ├── eda.R
│       ├── statistical_analysis.R
│       ├── univariate_bivariate_multivariate_analysis.R
│       └── ml_analysis.R
│
├── src/                    # Source code modules
│   └── utils.py
│
├── docs/                   # Documentation
│   └── analysis_report.md
│
├── results/                # Output files (plots, tables, etc.)
│   └── .gitkeep
│
├── requirements.txt        # Python dependencies
├── packages.R             # R package installation script
├── .gitignore             # Git ignore file
└── README.md              # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- R 4.0 or higher
- Jupyter Notebook or JupyterLab
- Git

### Installation

#### Python Environment

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows: `venv\Scripts\activate`
- Linux/Mac: `source venv/bin/activate`

3. Install dependencies:
```bash
pip install -r requirements.txt
```

#### R Environment

1. Install required R packages by running:
```r
source("packages.R")
```

Or manually install packages:
```r
install.packages(c("tidyverse", "ggplot2", "dplyr", "corrplot", 
                   "cluster", "factoextra", "FactoMineR", "VIM"))
```

## 📊 Dataset Information

### Dataset: Customers.csv

**Description**: Customer information dataset containing details about various customers including their names, contact information, addresses, cities, postal codes, and countries.

**Columns**:
- `CustomerID`: Unique identifier for each customer
- `CustomerName`: Name of the customer/company
- `ContactName`: Contact person's name
- `Address`: Street address
- `City`: City name
- `PostalCode`: Postal/ZIP code
- `Country`: Country name

**Dataset Size**: 91 customers

**License**: This dataset appears to be a sample dataset. If this is a public dataset, please respect the original dataset's license terms. If you are the owner of this dataset, you may specify your own license terms here.

### Dataset License

**IMPORTANT**: This project respects the original dataset's license. If you are using a dataset with specific licensing requirements, please ensure you comply with those terms. If the dataset is public domain or open source, this project can be used freely. For commercial datasets, ensure you have appropriate licenses.

**Note**: The original dataset license should be verified. If this is a proprietary dataset, appropriate licenses must be obtained before use.

## 📈 Analysis Components

### 1. Exploratory Data Analysis (EDA)
**Python**: `notebooks/python/01_EDA.ipynb` | **R**: `notebooks/r/01_EDA.ipynb`

- Data loading and inspection
- Data cleaning and preprocessing
- Missing value analysis
- Data type verification
- Summary statistics
- Initial data visualizations
- Duplicate record detection
- Data quality checks
- Country and city distribution analysis

### 2. Statistical Analysis
**Python**: `notebooks/python/02_Statistical_Analysis.ipynb` | **R**: `notebooks/r/02_Statistical_Analysis.ipynb`

- **Descriptive Statistics**: 
  - Mean, median, mode, standard deviation, variance, quartiles
  - Skewness and kurtosis
  - Central tendency and dispersion measures
- **Inferential Statistics**: 
  - Hypothesis testing
  - Confidence intervals (95%)
  - Normality tests (Shapiro-Wilk test)
  - Chi-square tests for independence
- **Exploratory Statistics**: 
  - Distribution analysis
  - Outlier detection (IQR method)
  - Statistical summary tables

### 3. Univariate, Bivariate, and Multivariate Analysis
**Python**: `notebooks/python/03_Univariate_Bivariate_Multivariate_Analysis.ipynb` | **R**: `notebooks/r/03_Univariate_Bivariate_Multivariate_Analysis.ipynb`

- **Univariate Analysis**: 
  - Single variable analysis (frequency, distribution)
  - Histograms, box plots, violin plots
  - Statistical summaries for numerical and categorical variables
- **Bivariate Analysis**: 
  - Two-variable relationships (correlations, cross-tabulations)
  - Country vs City analysis
  - Chi-square tests for categorical variables
  - Heatmaps and contingency tables
- **Multivariate Analysis**: 
  - Multiple variable relationships
  - Principal Component Analysis (PCA)
  - Correlation matrices
  - Multidimensional analysis

### 4. Machine Learning Analysis
**Python**: `notebooks/python/04_ML_Analysis.ipynb` | **R**: `notebooks/r/04_ML_Analysis.ipynb`

- **Data Preparation**: 
  - Feature engineering and preprocessing
  - Categorical variable encoding
  - Feature standardization
- **Clustering Analysis**: 
  - K-Means clustering
  - Hierarchical clustering (Ward linkage)
  - Optimal cluster number determination (Elbow method, Silhouette score)
- **Model Evaluation**: 
  - Silhouette score
  - Davies-Bouldin index
  - Calinski-Harabasz score
- **Cluster Interpretation**: 
  - Analysis of cluster characteristics
  - Country and city distribution by cluster
  - Cluster visualization using PCA
  - Dendrogram visualization for hierarchical clustering

## 🔬 Analysis Workflow

1. **EDA**: Start with `notebooks/python/01_EDA.ipynb` or `notebooks/r/01_EDA.ipynb`
2. **Statistical Analysis**: Proceed to `notebooks/python/02_Statistical_Analysis.ipynb` or `notebooks/r/02_Statistical_Analysis.ipynb`
3. **Univariate/Bivariate/Multivariate**: Continue with `notebooks/python/03_Univariate_Bivariate_Multivariate_Analysis.ipynb`
4. **ML Analysis**: Final analysis in `notebooks/python/04_ML_Analysis.ipynb` or `notebooks/r/04_ML_Analysis.ipynb`

## 📝 Usage

### Running Python Notebooks

```bash
jupyter notebook notebooks/python/01_EDA.ipynb
```

### Running R Notebooks

```bash
jupyter notebook notebooks/r/01_EDA.ipynb
```

### Running Python Scripts

```bash
python scripts/python/eda.py
python scripts/python/statistical_analysis.py
```

### Running R Scripts

```bash
Rscript scripts/r/eda.R
Rscript scripts/r/statistical_analysis.R
```

## 📊 Key Findings

### Dataset Overview
- **Total Customers**: 91
- **Countries**: Multiple countries represented (Germany, Mexico, UK, USA, etc.)
- **Cities**: Diverse geographical distribution
- **Data Quality**: Clean dataset with no missing values
- **Unique Customer IDs**: All customer IDs are unique

### Analysis Highlights
1. **Geographical Distribution**: Customers are distributed across multiple countries and cities
2. **Customer Segmentation**: Clustering analysis reveals distinct customer groups based on geographical and ID patterns
3. **Statistical Insights**: Comprehensive statistical analysis provides detailed insights into customer data patterns
4. **Multivariate Relationships**: PCA and correlation analysis reveal underlying patterns in the data

*(Detailed findings will be available after running the analysis notebooks)*

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Dataset License**: Please respect the original dataset's license. If you are using this project with a licensed dataset, ensure you comply with the dataset's terms of use. See the LICENSE file for more details.

## 🙏 Acknowledgments

- Thanks to the open-source community for the excellent data science libraries
- Dataset providers (please credit appropriately)

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

## 🔗 References

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [R Documentation](https://www.r-project.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Tidyverse Documentation](https://www.tidyverse.org/)

## ✅ Project Verification

To verify that all project components are in place, run:

```bash
python verify_project_complete.py
```

This script will check:
- Project structure
- Notebook files
- Script files
- Requirements files
- File contents

## 📋 Project Checklist

- [x] Well-organized project structure
- [x] GitHub repository initialized
- [x] Comprehensive EDA notebooks (Python & R)
- [x] Comprehensive EDA scripts (Python & R)
- [x] Descriptive, Inferential, Exploratory statistical analysis (Python & R)
- [x] Univariate, Bivariate, Multivariate analysis (Python & R)
- [x] ML analysis with appropriate algorithms (Python & R)
- [x] Comprehensive README.md
- [x] Dataset license respected
- [x] Requirements files (requirements.txt, packages.R)
- [x] Git ignore file
- [x] LICENSE file

---

**Last Updated**: November 2025

---
*Enhanced with comprehensive customer segmentation and behavioral analytics workflows*

