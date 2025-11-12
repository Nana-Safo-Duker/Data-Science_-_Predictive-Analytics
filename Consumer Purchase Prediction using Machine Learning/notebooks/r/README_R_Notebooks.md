# R Notebooks Guide

This directory contains R notebooks for comprehensive data analysis. Due to the complexity of creating R Jupyter notebooks programmatically, we provide R scripts in the `scripts/r/` directory that contain all the analysis code.

## Using R Notebooks

### Option 1: Convert R Scripts to Notebooks

You can convert the R scripts to Jupyter notebooks using the following approach:

1. Install IRkernel in R:
```R
install.packages("IRkernel")
IRkernel::installspec()
```

2. Create a new R notebook in Jupyter:
   - Open Jupyter Notebook
   - Select "New" -> "R"
   - Copy code from the corresponding R script in `scripts/r/`
   - Run the cells

### Option 2: Use R Markdown

Alternatively, you can create R Markdown notebooks (`.Rmd`) which are more suitable for R:

1. Create an R Markdown file (`.Rmd`)
2. Use the code from the R scripts
3. Knit to HTML or PDF

### Available R Scripts

- `scripts/r/eda.R` - Exploratory Data Analysis
- `scripts/r/statistical_analysis.R` - Statistical Analysis
- `scripts/r/univariate_bivariate_multivariate.R` - Variable Analysis
- `scripts/r/ml_analysis.R` - Machine Learning Analysis

## Running R Scripts

You can run the R scripts directly:

```R
# In R or RStudio
source("scripts/r/eda.R")
source("scripts/r/statistical_analysis.R")
source("scripts/r/univariate_bivariate_multivariate.R")
source("scripts/r/ml_analysis.R")
```

## Notebook Structure

Each notebook should follow this structure:

1. **Setup and Data Loading**
   - Load libraries
   - Load dataset
   - Basic data overview

2. **Analysis Sections**
   - EDA: Exploratory data analysis
   - Statistical Analysis: Descriptive and inferential statistics
   - Univariate/Bivariate/Multivariate Analysis
   - Machine Learning: Model training and evaluation

3. **Results and Visualizations**
   - Generate plots
   - Display results
   - Save outputs

## Notes

- All R scripts are fully functional and can be run independently
- Output files are saved to the `output/` directory
- Make sure to install all required R packages before running
- Adjust file paths if needed based on your working directory

