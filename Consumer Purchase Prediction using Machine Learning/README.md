# Consumer Purchase Prediction

A comprehensive data science project for predicting consumer purchase behavior using machine learning techniques. This project includes exploratory data analysis, statistical analysis, and machine learning models implemented in both Python and R.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [License](#license)
- [Contributing](#contributing)

## 🎯 Project Overview

This project aims to predict whether a consumer will purchase a product based on demographic and socioeconomic factors such as age, gender, and estimated salary. The project includes:

- **Exploratory Data Analysis (EDA)**: Comprehensive analysis of the dataset
- **Statistical Analysis**: Descriptive, inferential, and exploratory statistics
- **Univariate, Bivariate, and Multivariate Analysis**: Detailed analysis of variable relationships
- **Machine Learning Models**: Multiple algorithms for prediction including:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - Naive Bayes
  - Decision Tree
  - Gradient Boosting

## 📊 Dataset

The dataset contains information about consumers including:
- **User ID**: Unique identifier for each user
- **Gender**: Male or Female
- **Age**: Age of the consumer
- **EstimatedSalary**: Estimated salary of the consumer
- **Purchased**: Target variable (0 = No Purchase, 1 = Purchase)

### Dataset License

This dataset is used for educational and research purposes. Please respect the original dataset's license and terms of use. If you are the original creator of this dataset and wish to specify the license, please contact the project maintainers.

## 📁 Project Structure

```
Consumer Purchase Prediction/
├── data/
│   ├── Advertisement.csv          # Main dataset
│   └── .gitkeep
├── notebooks/
│   ├── python/
│   │   ├── 01_EDA_Python.ipynb
│   │   ├── 02_Statistical_Analysis_Python.ipynb
│   │   ├── 03_Univariate_Bivariate_Multivariate_Analysis.ipynb
│   │   └── 04_ML_Analysis_Python.ipynb
│   └── r/
│       └── README_R_Notebooks.md    # Guide for creating R notebooks
├── scripts/
│   ├── python/
│   │   ├── eda.py
│   │   ├── univariate_bivariate_multivariate.py
│   │   └── ml_analysis.py
│   └── r/
│       ├── eda.R
│       ├── statistical_analysis.R
│       ├── univariate_bivariate_multivariate.R
│       └── ml_analysis.R
├── requirements/
│   ├── requirements.txt           # Python dependencies
│   └── requirements_r.txt         # R dependencies
├── output/                        # Generated plots and visualizations
├── models/                        # Saved machine learning models
├── documentation/                 # Additional documentation
├── .gitignore
├── LICENSE                        # MIT License
└── README.md

```

## ✨ Features

### Python Implementation
- **EDA**: Comprehensive exploratory data analysis with visualizations
- **Statistical Analysis**: Descriptive statistics, hypothesis testing, correlation analysis, ANOVA
- **Univariate Analysis**: Individual variable analysis with distributions, Q-Q plots, and statistics
- **Bivariate Analysis**: Relationship analysis between pairs of variables
- **Multivariate Analysis**: Comprehensive analysis of multiple variables simultaneously
- **Machine Learning**: Multiple algorithms with cross-validation and performance metrics

### R Implementation
- **EDA**: Exploratory data analysis using R's statistical and visualization packages
- **Statistical Analysis**: Comprehensive statistical tests and analysis
- **Univariate, Bivariate, Multivariate Analysis**: Detailed variable relationship analysis
- **Machine Learning**: ML models using caret and other R packages

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- R 4.0 or higher (for R notebooks)
- Jupyter Notebook or JupyterLab
- Git

### Python Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd "Consumer Purchase Prediction"
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements/requirements.txt
```

### R Setup

1. Install R packages:
```R
install.packages(c("dplyr", "ggplot2", "corrplot", "caret", "randomForest", 
                   "e1071", "rpart", "rpart.plot", "pROC", "ROCR", "knitr", 
                   "rmarkdown"))
```

Or install from the requirements file:
```R
packages <- readLines("requirements/requirements_r.txt")
install.packages(packages)
```

2. Install IRkernel for Jupyter (optional):
```R
install.packages("IRkernel")
IRkernel::installspec()
```

## 📖 Usage

### Python Notebooks

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Navigate to `notebooks/python/` and open the desired notebook:
   - `01_EDA_Python.ipynb`: Exploratory Data Analysis
   - `02_Statistical_Analysis_Python.ipynb`: Statistical Analysis
   - `03_Univariate_Bivariate_Multivariate_Analysis.ipynb`: Variable Analysis
   - `04_ML_Analysis_Python.ipynb`: Machine Learning Models

### Python Scripts

Run scripts directly from the command line:

```bash
# EDA
python scripts/python/eda.py

# Univariate, Bivariate, Multivariate Analysis
python scripts/python/univariate_bivariate_multivariate.py

# Machine Learning Analysis
python scripts/python/ml_analysis.py
```

### R Notebooks

1. Start Jupyter Notebook with R kernel:
```bash
jupyter notebook
```

2. Navigate to `notebooks/r/` and open the desired notebook

### R Scripts

Run R scripts from the command line or RStudio. The scripts automatically detect the project root and data files, so you can run them from any directory:

```R
# From R or RStudio - scripts will auto-detect paths
source("Consumer Purchase Prediction/Consumer Purchase Prediction/scripts/r/eda.R")
source("Consumer Purchase Prediction/Consumer Purchase Prediction/scripts/r/statistical_analysis.R")
source("Consumer Purchase Prediction/Consumer Purchase Prediction/scripts/r/univariate_bivariate_multivariate.R")
source("Consumer Purchase Prediction/Consumer Purchase Prediction/scripts/r/ml_analysis.R")
```

Or from the command line:
```bash
Rscript "Consumer Purchase Prediction/Consumer Purchase Prediction/scripts/r/eda.R"
```

**Note:** The scripts automatically find the project root by searching for the `data/Advertisement.csv` file. If you encounter path errors, make sure you're running the scripts from a directory that contains the project folder, or set your working directory to the workspace root.

## 📈 Results

### Key Findings

1. **Dataset Characteristics**:
   - Total observations: 401
   - Features: Gender, Age, EstimatedSalary
   - Target: Purchased (Binary classification)

2. **Statistical Insights**:
   - Age and Estimated Salary show significant differences between purchase and non-purchase groups
   - Strong correlation between age and purchase behavior
   - Gender may have some influence on purchase decisions

3. **Machine Learning Performance**:
   - Multiple models were evaluated
   - Best performing model varies based on metrics
   - Cross-validation ensures model reliability

### Model Performance

The project evaluates multiple machine learning algorithms:
- **Logistic Regression**: Baseline model for binary classification
- **Random Forest**: Ensemble method with good interpretability
- **SVM**: Support Vector Machine for classification
- **Naive Bayes**: Probabilistic classifier
- **Decision Tree**: Interpretable tree-based model
- **Gradient Boosting**: Advanced ensemble method

Performance metrics include:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Cross-validation scores

## 🛠️ Technologies Used

### Python
- **Data Manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Statistical Analysis**: scipy, statsmodels
- **Machine Learning**: scikit-learn
- **Jupyter**: jupyter, ipykernel

### R
- **Data Manipulation**: dplyr, tidyr, readr
- **Visualization**: ggplot2, plotly, corrplot
- **Statistical Analysis**: stats, car, psych
- **Machine Learning**: caret, randomForest, e1071, rpart
- **Utilities**: knitr, rmarkdown

## 📝 License

This project is provided for educational and research purposes. Please respect the original dataset's license. If you use this project or dataset, please:

1. Acknowledge the original data source
2. Follow any licensing terms specified by the dataset creator
3. Use the data responsibly and ethically

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Contributing Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

## 🙏 Acknowledgments

- Dataset creators and providers
- Open-source community for excellent tools and libraries
- Contributors and reviewers

## 📚 References

- Scikit-learn Documentation: https://scikit-learn.org/
- R Documentation: https://www.r-project.org/
- Pandas Documentation: https://pandas.pydata.org/
- Matplotlib Documentation: https://matplotlib.org/

---

**Note**: This project is for educational purposes. Always ensure you have the right to use any dataset and comply with relevant data protection regulations.

