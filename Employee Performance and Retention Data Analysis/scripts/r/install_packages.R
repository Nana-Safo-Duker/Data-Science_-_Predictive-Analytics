# Install Required R Packages for Employee Dataset Analysis
# Run this script to install all required packages

# List of required packages
required_packages <- c(
  "tidyverse",      # Data manipulation and visualization (includes dplyr, tidyr, ggplot2)
  "ggplot2",        # Advanced plotting
  "dplyr",          # Data manipulation
  "tidyr",          # Data tidying (pivot_wider, etc.)
  "lubridate",      # Date handling
  "VIM",            # Missing data visualization
  "corrplot",       # Correlation plots
  "psych",          # Psychological research tools (descriptive statistics)
  "car",            # Companion to Applied Regression
  "nortest",        # Normality tests
  "caret",          # Classification and regression training
  "randomForest",   # Random forest
  "xgboost",        # XGBoost
  "e1071",          # Misc functions (SVM, etc.)
  "plotly",         # Interactive plots
  "gridExtra",      # Grid arrangements
  "RColorBrewer",   # Color palettes
  "GGally"          # Extension to ggplot2
)

# Function to install packages if not already installed
install_if_missing <- function(package) {
  if (!require(package, character.only = TRUE)) {
    install.packages(package, dependencies = TRUE)
    library(package, character.only = TRUE)
  }
}

# Install all required packages
cat("Installing required R packages...\n")
for (pkg in required_packages) {
  cat("Checking/Installing", pkg, "...\n")
  install_if_missing(pkg)
}

cat("\nAll required packages have been installed successfully!\n")
cat("You can now run the analysis scripts.\n")
