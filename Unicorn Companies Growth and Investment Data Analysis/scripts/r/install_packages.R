# Install required R packages for Unicorn Companies Analysis

# Core data manipulation and visualization
install.packages(c("dplyr", "ggplot2", "tidyr", "readr", "stringr", "lubridate"))

# Statistical analysis
install.packages(c("corrplot", "VIM", "e1071", "broom"))

# Machine learning
install.packages(c("caret", "randomForest", "xgboost", "nnet", "e1071"))

# Visualization and reporting
install.packages(c("GGally", "gridExtra", "plotly"))

# Documentation
install.packages(c("knitr", "rmarkdown"))

# Check if all packages are installed
required_packages <- c("dplyr", "ggplot2", "tidyr", "readr", "stringr", "lubridate",
                      "corrplot", "VIM", "e1071", "broom",
                      "caret", "randomForest", "xgboost", "nnet",
                      "GGally", "gridExtra", "plotly",
                      "knitr", "rmarkdown")

missing_packages <- required_packages[!required_packages %in% installed.packages()[,"Package"]]

if(length(missing_packages) > 0) {
  cat("Missing packages:", paste(missing_packages, collapse = ", "), "\n")
  cat("Please install them using: install.packages(c(", 
      paste(paste0('"', missing_packages, '"'), collapse = ", "), "))\n")
} else {
  cat("All required packages are installed!\n")
}


