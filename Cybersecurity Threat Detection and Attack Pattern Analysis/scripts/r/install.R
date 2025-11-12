# Install required R packages for Cybersecurity Attacks Analysis

# List of required packages
required_packages <- c(
  # Data Manipulation
  "data.table",
  "dplyr",
  "tidyr",
  
  # Data Visualization
  "ggplot2",
  "plotly",
  "corrplot",
  "VIM",
  "naniar",
  
  # Statistical Analysis
  "stats",
  "psych",
  "car",
  "DescTools",
  
  # Machine Learning
  "randomForest",
  "e1071",
  "caret",
  "rpart",
  "rpart.plot",
  "xgboost",
  "glmnet",
  
  # Jupyter
  "IRkernel",
  "repr",
  "IRdisplay",
  
  # Utilities
  "readr",
  "stringr",
  "lubridate"
)

# Function to install packages if not already installed
install_if_missing <- function(packages) {
  new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
  if(length(new_packages)) {
    install.packages(new_packages, dependencies = TRUE)
  }
  # Load all packages
  lapply(packages, require, character.only = TRUE)
}

# Install and load packages
install_if_missing(required_packages)

cat("All required R packages have been installed and loaded successfully!\n")



