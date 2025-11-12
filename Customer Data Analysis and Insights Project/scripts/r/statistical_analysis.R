# Statistical Analysis Script
# Descriptive, Inferential, and Exploratory Statistical Analysis

# Load libraries
library(tidyverse)
library(ggplot2)
library(dplyr)
library(corrplot)

# Set paths
data_path <- "../../data/Customers.csv"
results_path <- "../../results"

# Create results directory if it doesn't exist
if (!dir.exists(results_path)) {
  dir.create(results_path, recursive = TRUE)
}

# Load data
df <- read.csv(data_path, stringsAsFactors = FALSE)

cat("==================================================\n")
cat("STATISTICAL ANALYSIS\n")
cat("==================================================\n\n")

# 1. Descriptive Statistics
cat("=== DESCRIPTIVE STATISTICS ===\n\n")

# Numerical variables
numerical_cols <- sapply(df, is.numeric)
if (sum(numerical_cols) > 0) {
  cat("=== Numerical Variables Descriptive Statistics ===\n")
  print(summary(df[, numerical_cols, drop = FALSE]))
  
  # Additional descriptive statistics
  cat("\n=== Additional Descriptive Statistics ===\n")
  desc_stats <- df[, numerical_cols, drop = FALSE] %>%
    summarise_all(list(
      Mean = ~mean(., na.rm = TRUE),
      Median = ~median(., na.rm = TRUE),
      StdDev = ~sd(., na.rm = TRUE),
      Variance = ~var(., na.rm = TRUE),
      Min = ~min(., na.rm = TRUE),
      Max = ~max(., na.rm = TRUE),
      Q1 = ~quantile(., 0.25, na.rm = TRUE),
      Q3 = ~quantile(., 0.75, na.rm = TRUE),
      IQR = ~IQR(., na.rm = TRUE)
    ))
  print(desc_stats)
} else {
  cat("No numerical columns found.\n")
}

# Categorical variables
cat("\n=== Categorical Variables Descriptive Statistics ===\n")
categorical_cols <- sapply(df, is.character) | sapply(df, is.factor)
for (col in names(df)[categorical_cols]) {
  cat("\n", col, ":\n", sep = "")
  cat("  Count:", nrow(df), "\n")
  cat("  Unique values:", n_distinct(df[[col]]), "\n")
  cat("  Mode:", names(sort(table(df[[col]]), decreasing = TRUE)[1]), "\n")
  cat("  Mode frequency:", max(table(df[[col]])), "\n")
  cat("  Mode percentage:", round((max(table(df[[col]])) / nrow(df)) * 100, 2), "%\n")
}

# 2. Inferential Statistics
cat("\n==================================================\n")
cat("INFERENTIAL STATISTICS\n")
cat("==================================================\n\n")

# Confidence Intervals
if (sum(numerical_cols) > 0) {
  cat("=== Confidence Intervals (95%) ===\n")
  for (col in names(df)[numerical_cols]) {
    data <- df[[col]][!is.na(df[[col]])]
    mean_val <- mean(data)
    std_val <- sd(data)
    n <- length(data)
    
    margin_error <- 1.96 * (std_val / sqrt(n))
    ci_lower <- mean_val - margin_error
    ci_upper <- mean_val + margin_error
    
    cat("\n", col, ":\n", sep = "")
    cat("  Mean:", round(mean_val, 2), "\n")
    cat("  95% CI: [", round(ci_lower, 2), ", ", round(ci_upper, 2), "]\n", sep = "")
  }
}

# Normality Tests
if (sum(numerical_cols) > 0) {
  cat("\n=== Normality Tests ===\n")
  alpha <- 0.05
  for (col in names(df)[numerical_cols]) {
    data <- df[[col]][!is.na(df[[col]])]
    
    # Shapiro-Wilk test (for small samples)
    if (length(data) <= 5000) {
      shapiro_test <- shapiro.test(data)
      cat("\n", col, ":\n", sep = "")
      cat("  Shapiro-Wilk test:\n")
      cat("    Statistic:", round(shapiro_test$statistic, 4), "\n")
      cat("    p-value:", round(shapiro_test$p.value, 4), "\n")
      
      if (shapiro_test$p.value > alpha) {
        cat("    Result: Data appears to be normally distributed (p > ", alpha, ")\n", sep = "")
      } else {
        cat("    Result: Data does not appear to be normally distributed (p <= ", alpha, ")\n", sep = "")
      }
    }
  }
}

# Chi-Square Test for Independence
cat("\n=== Chi-Square Test for Independence ===\n")
if ("Country" %in% colnames(df) && "City" %in% colnames(df)) {
  contingency_table <- table(df$Country, df$City)
  chi2_test <- chisq.test(contingency_table)
  
  cat("\nChi-square statistic:", round(chi2_test$statistic, 4), "\n")
  cat("Degrees of freedom:", chi2_test$parameter, "\n")
  cat("p-value:", round(chi2_test$p.value, 4), "\n")
  
  alpha <- 0.05
  if (chi2_test$p.value < alpha) {
    cat("\nResult: Reject null hypothesis. Country and City are not independent (p < ", alpha, ")\n", sep = "")
  } else {
    cat("\nResult: Fail to reject null hypothesis. Country and City may be independent (p >= ", alpha, ")\n", sep = "")
  }
}

# 3. Exploratory Statistical Analysis
cat("\n==================================================\n")
cat("EXPLORATORY STATISTICAL ANALYSIS\n")
cat("==================================================\n\n")

# Distribution Analysis
if (sum(numerical_cols) > 0) {
  cat("=== Distribution Analysis ===\n")
  for (col in names(df)[numerical_cols]) {
    # Histogram
    png(file = file.path(results_path, paste0(col, "_histogram.png")), 
        width = 1400, height = 600, res = 300)
    hist(df[[col]], breaks = 20, main = paste("Distribution of", col),
         xlab = col, ylab = "Frequency", col = "steelblue", border = "black")
    dev.off()
    
    # Box plot
    png(file = file.path(results_path, paste0(col, "_boxplot.png")), 
        width = 800, height = 600, res = 300)
    boxplot(df[[col]], main = paste("Box Plot of", col), ylab = col, col = "lightblue")
    dev.off()
    
    cat("✓ Plots saved for", col, "\n")
  }
}

# Outlier Detection
if (sum(numerical_cols) > 0) {
  cat("\n=== Outlier Detection (IQR Method) ===\n")
  for (col in names(df)[numerical_cols]) {
    Q1 <- quantile(df[[col]], 0.25, na.rm = TRUE)
    Q3 <- quantile(df[[col]], 0.75, na.rm = TRUE)
    IQR_val <- Q3 - Q1
    
    lower_bound <- Q1 - 1.5 * IQR_val
    upper_bound <- Q3 + 1.5 * IQR_val
    
    outliers <- df[[col]][df[[col]] < lower_bound | df[[col]] > upper_bound]
    
    cat("\n", col, ":\n", sep = "")
    cat("  Lower bound:", round(lower_bound, 2), "\n")
    cat("  Upper bound:", round(upper_bound, 2), "\n")
    cat("  Number of outliers:", length(outliers), "\n")
    if (length(outliers) > 0) {
      cat("  Outlier values:", paste(outliers, collapse = ", "), "\n")
    }
  }
}

# Central Tendency and Dispersion
if (sum(numerical_cols) > 0) {
  cat("\n=== Central Tendency and Dispersion Measures ===\n")
  measures <- df[, numerical_cols, drop = FALSE] %>%
    summarise_all(list(
      Mean = ~mean(., na.rm = TRUE),
      Median = ~median(., na.rm = TRUE),
      StdDev = ~sd(., na.rm = TRUE),
      Variance = ~var(., na.rm = TRUE),
      CV = ~(sd(., na.rm = TRUE) / mean(., na.rm = TRUE)) * 100
    ))
  print(measures)
}

cat("\n==================================================\n")
cat("STATISTICAL ANALYSIS SUMMARY\n")
cat("==================================================\n")
cat("\n1. Descriptive Statistics:\n")
cat("   • Calculated measures of central tendency (mean, median, mode)\n")
cat("   • Calculated measures of dispersion (std dev, variance, IQR)\n")
cat("   • Analyzed distribution characteristics\n")
cat("\n2. Inferential Statistics:\n")
cat("   • Calculated 95% confidence intervals\n")
cat("   • Performed normality tests\n")
cat("   • Conducted chi-square tests for independence\n")
cat("\n3. Exploratory Statistics:\n")
cat("   • Analyzed distributions\n")
cat("   • Detected outliers\n")
cat("   • Examined central tendency and dispersion\n")

cat("\n✓ Statistical analysis completed successfully!\n")

