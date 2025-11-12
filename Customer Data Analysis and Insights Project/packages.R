# R Package Installation Script
# Customer Data Analysis Project

# List of required R packages
required_packages <- c(
  # Data Manipulation
  "tidyverse",
  "dplyr",
  "tidyr",
  
  # Visualization
  "ggplot2",
  "corrplot",
  
  # Statistical Analysis
  "stats",
  
  # Machine Learning
  "cluster",
  "factoextra",
  "FactoMineR",
  
  # Additional utilities
  "VIM"
)

# Function to check and install packages
install_if_missing <- function(packages) {
  new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
  if(length(new_packages)) {
    install.packages(new_packages, dependencies = TRUE)
  }
  # Load all packages
  for(pkg in packages) {
    if(!require(pkg, character.only = TRUE)) {
      stop(paste("Package", pkg, "could not be loaded"))
    }
  }
}

# Install and load packages
cat("Installing and loading required R packages...\n")
install_if_missing(required_packages)
cat("All packages installed and loaded successfully!\n")

# Verify installation
cat("\n=== Installed Packages ===\n")
for(pkg in required_packages) {
  if(require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat("✓", pkg, "\n")
  } else {
    cat("✗", pkg, "(failed to load)\n")
  }
}

cat("\n=== Package Installation Complete ===\n")


