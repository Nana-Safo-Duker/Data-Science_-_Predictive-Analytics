# Email Spam Detection - Comprehensive Data Science Project

## 📋 Project Overview

This project provides a comprehensive analysis of an email spam detection dataset, including Exploratory Data Analysis (EDA), statistical analysis, and machine learning models to classify emails as spam or ham (non-spam).

## 🎯 Objectives

1. Perform comprehensive Exploratory Data Analysis (EDA)
2. Conduct descriptive, inferential, and exploratory statistical analysis
3. Perform univariate, bivariate, and multivariate analysis
4. Build and evaluate machine learning models for spam detection
5. Compare model performance and identify the best algorithm

## 📊 Dataset

- **Source**: Email Spam Dataset
- **Size**: 5,726 emails
- **Features**: 
  - `text`: Email content (text)
  - `spam`: Binary label (0 = Ham, 1 = Spam)
- **Class Distribution**:
  - Ham (0): 4,358 emails (76.1%)
  - Spam (1): 1,368 emails (23.9%)

## 📁 Project Structure

```
emails-spam/
│
├── data/
│   ├── emails-spam.csv              # Original dataset
│   └── emails_spam_clean.csv        # Cleaned dataset
│
├── notebooks/
│   ├── python/
│   │   ├── 01_EDA_Python.ipynb
│   │   ├── 02_Statistical_Analysis_Python.ipynb
│   │   ├── 03_Univariate_Bivariate_Multivariate_Analysis.ipynb
│   │   └── 04_ML_Analysis_Python.ipynb
│   └── R/
│       ├── 01_EDA_R.ipynb
│       ├── 02_Statistical_Analysis_R.ipynb
│       ├── 03_Univariate_Bivariate_Multivariate_Analysis_R.ipynb
│       └── 04_ML_Analysis_R.ipynb
│
├── scripts/
│   ├── python/
│   │   ├── eda.py
│   │   ├── statistical_analysis.py
│   │   ├── univariate_bivariate_multivariate.py
│   │   └── ml_analysis.py
│   └── R/
│       ├── eda.R
│       ├── statistical_analysis.R
│       ├── univariate_bivariate_multivariate.R
│       └── ml_analysis.R
│
├── models/                           # Saved models
├── output/
│   └── figures/                      # Generated visualizations
│
├── requirements.txt                  # Python dependencies
├── requirements_r.txt                # R package list
├── .gitignore
└── README.md

```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- R 4.0+
- Jupyter Notebook
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd emails-spam
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install R packages**
   ```r
   # Open R and run:
   install.packages(c("tidyverse", "caret", "tm", "ggplot2", "e1071", "randomForest", "xgboost", "naivebayes"))
   ```

4. **Setup Jupyter Kernel for R** (optional)
   ```r
   install.packages('IRkernel')
   IRkernel::installspec()
   ```

## 📓 Notebooks Description

### Python Notebooks

1. **01_EDA_Python.ipynb**: Comprehensive Exploratory Data Analysis
   - Data loading and cleaning
   - Basic statistics
   - Text preprocessing
   - Visualizations
   - Feature engineering

2. **02_Statistical_Analysis_Python.ipynb**: Statistical Analysis
   - Descriptive statistics
   - Inferential statistics
   - Hypothesis testing
   - Correlation analysis

3. **03_Univariate_Bivariate_Multivariate_Analysis.ipynb**: Variable Analysis
   - Univariate analysis
   - Bivariate analysis
   - Multivariate analysis
   - Feature relationships

4. **04_ML_Analysis_Python.ipynb**: Machine Learning Models
   - Data preprocessing
   - Feature extraction (TF-IDF, Bag of Words)
   - Model training and evaluation
   - Algorithms: Naive Bayes, SVM, Random Forest, XGBoost, Logistic Regression

### R Notebooks

1. **01_EDA_R.ipynb**: Exploratory Data Analysis in R
2. **02_Statistical_Analysis_R.ipynb**: Statistical Analysis in R
3. **03_Univariate_Bivariate_Multivariate_Analysis_R.ipynb**: Variable Analysis in R
4. **04_ML_Analysis_R.ipynb**: Machine Learning Models in R

## 🔬 Methodology

### Data Preprocessing
- Text cleaning and normalization
- Removal of special characters
- Tokenization
- Stop word removal
- Stemming/Lemmatization

### Feature Engineering
- Text length features
- Word count features
- Character count features
- TF-IDF vectorization
- Bag of Words

### Machine Learning Models
- Naive Bayes
- Support Vector Machine (SVM)
- Random Forest
- XGBoost
- Logistic Regression

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

## 📈 Key Findings

After running the analysis, you will find:

1. **Dataset Characteristics**:
   - Total emails: 5,726
   - Spam emails: 1,368 (23.9%)
   - Ham emails: 4,358 (76.1%)

2. **Text Statistics**:
   - Average text length varies between spam and ham emails
   - Word count distributions show distinct patterns
   - Character-level features can be discriminative

3. **Machine Learning Performance**:
   - Multiple algorithms evaluated (Naive Bayes, SVM, Random Forest, XGBoost, Logistic Regression)
   - Best model selected based on F1-score
   - Performance metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

## 🎯 Key Insights

(To be updated after running the analysis)

## 📝 License

### Dataset License
This dataset is provided for educational and research purposes. Please respect the original dataset's license terms. If you use this dataset, please cite the original source appropriately.

**Important**: 
- The dataset is used here for educational and research purposes only
- Please verify and comply with the original dataset's terms of use
- If you plan to use this dataset for commercial purposes, ensure you have the proper licenses
- Always cite the original dataset source when using this data

### Project License
This project is open source and available under the MIT License. See the LICENSE file for details.

### Usage Guidelines
1. **Educational Use**: This project is intended for educational purposes
2. **Research Use**: Suitable for academic research and learning
3. **Commercial Use**: Verify dataset licensing before commercial use
4. **Attribution**: Please provide appropriate attribution when using this project

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👥 Authors

- Project Contributors

## 📊 Dataset Information

### Dataset Characteristics
- **Total Samples**: 5,726 emails
- **Features**: Text content, spam label
- **Target Variable**: Binary classification (0 = Ham, 1 = Spam)
- **Class Distribution**: Imbalanced (76.1% Ham, 23.9% Spam)

### Data Preprocessing
- Text cleaning and normalization
- Removal of URLs and email addresses
- Special character handling
- Feature engineering (text length, word count, etc.)

### Model Performance
Models are evaluated using:
- Cross-validation (5-fold)
- Multiple metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- Confusion matrices
- ROC curves

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

## 🙏 Acknowledgments

- Original dataset creators
- Open-source community for tools and libraries

## 📚 References

- Scikit-learn documentation
- NLTK documentation
- R documentation
- Various machine learning and NLP resources

---

**Note**: This is a comprehensive data science project demonstrating various analysis techniques and machine learning approaches for email spam detection.

