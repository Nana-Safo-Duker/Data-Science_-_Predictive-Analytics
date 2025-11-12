# Statistical Analysis Script for Fraud Detection Dataset (R)
#
# This script performs comprehensive statistical analysis including:
# 1. Descriptive Statistics - Mean, median, mode, standard deviation, variance, skewness, kurtosis
# 2. Inferential Statistics - Hypothesis testing, confidence intervals, t-tests, chi-square tests
# 3. Exploratory Statistics - Correlation analysis, feature relationships, statistical tests

# Load libraries
library(tidyverse)
library(car)
library(psych)
library(corrplot)
library(ggplot2)

# Set options
options(warn = -1)
set.seed(42)

# Set up paths
project_root <- dirname(dirname(dirname(getwd())))
data_path <- file.path(project_root, "data", "fraud_data.csv")
output_dir <- file.path(project_root, "outputs", "figures")

# Create output directory if it doesn't exist
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Load data
cat("Loading data...\n")
df <- read.csv(data_path, stringsAsFactors = FALSE)
cat("Data loaded:", dim(df), "\n")
cat("Target variable distribution:\n")
print(table(df$isFraud))

# ============================================================================
# 1. DESCRIPTIVE STATISTICS
# ============================================================================
cat("\n", rep("=", 80), "\n", sep = "")
cat("DESCRIPTIVE STATISTICS\n")
cat(rep("=", 80), "\n", sep = "")

# Key features for analysis
key_features <- c("TransactionAmt", "card1", "card2", "card3", "card5")
key_features <- key_features[key_features %in% colnames(df)]

if(length(key_features) > 0) {
  # Descriptive statistics
  desc_stats <- psych::describe(df[key_features])
  cat("\nDescriptive Statistics for Key Numerical Features:\n")
  print(desc_stats)
  
  # Statistics by fraud status
  if("TransactionAmt" %in% key_features) {
    cat("\nTransaction Amount Statistics by Fraud Status:\n")
    fraud_stats <- df %>%
      group_by(isFraud) %>%
      summarise(
        Count = n(),
        Mean = mean(TransactionAmt, na.rm = TRUE),
        Median = median(TransactionAmt, na.rm = TRUE),
        Std = sd(TransactionAmt, na.rm = TRUE),
        Min = min(TransactionAmt, na.rm = TRUE),
        Max = max(TransactionAmt, na.rm = TRUE),
        Skewness = psych::skew(TransactionAmt, na.rm = TRUE),
        Kurtosis = psych::kurtosi(TransactionAmt, na.rm = TRUE),
        .groups = 'drop'
      )
    print(fraud_stats)
    
    # Visualization
    png(file.path(output_dir, "descriptive_stats_transaction_r.png"), 
        width = 1200, height = 600, res = 300)
    par(mfrow = c(1, 2))
    
    # Histogram
    hist(df$TransactionAmt[df$isFraud == 0], breaks = 50, col = rgb(0, 0, 1, 0.5),
         main = "Transaction Amount Distribution by Fraud Status",
         xlab = "Transaction Amount", ylab = "Frequency", xlim = c(0, quantile(df$TransactionAmt, 0.99, na.rm = TRUE)))
    hist(df$TransactionAmt[df$isFraud == 1], breaks = 50, col = rgb(1, 0, 0, 0.5), add = TRUE)
    legend("topright", legend = c("Legitimate", "Fraud"), fill = c(rgb(0, 0, 1, 0.5), rgb(1, 0, 0, 0.5)))
    
    # Box plot
    boxplot(TransactionAmt ~ isFraud, data = df, log = "y",
            main = "Transaction Amount by Fraud Status",
            xlab = "Fraud Status", ylab = "Transaction Amount (log scale)",
            names = c("Legitimate", "Fraud"))
    
    dev.off()
    cat("\nFigure saved to", file.path(output_dir, "descriptive_stats_transaction_r.png"), "\n")
  }
}

# ============================================================================
# 2. INFERENTIAL STATISTICS
# ============================================================================
cat("\n", rep("=", 80), "\n", sep = "")
cat("INFERENTIAL STATISTICS\n")
cat(rep("=", 80), "\n", sep = "")

# Hypothesis Test 1: Transaction Amount
if("TransactionAmt" %in% colnames(df)) {
  cat("\nHypothesis Test: Transaction Amount\n")
  cat("H0: Mean transaction amount is the same for fraud and legitimate transactions\n")
  cat("H1: Mean transaction amount differs between fraud and legitimate transactions\n\n")
  
  fraud_amt <- df$TransactionAmt[df$isFraud == 1 & !is.na(df$TransactionAmt)]
  legit_amt <- df$TransactionAmt[df$isFraud == 0 & !is.na(df$TransactionAmt)]
  
  # Mann-Whitney U test (non-parametric)
  test_result <- wilcox.test(fraud_amt, legit_amt, alternative = "two.sided")
  cat("Mann-Whitney U Test Results:\n")
  print(test_result)
  
  if(test_result$p.value < 0.05) {
    cat("Result: REJECT H0 - Mean transaction amounts are significantly different\n")
  } else {
    cat("Result: FAIL TO REJECT H0 - No significant difference in mean transaction amounts\n")
  }
  
  # Confidence intervals
  fraud_ci <- t.test(fraud_amt)$conf.int
  legit_ci <- t.test(legit_amt)$conf.int
  
  cat("\n95% Confidence Intervals:\n")
  cat("Fraud transactions:", fraud_ci[1], "-", fraud_ci[2], 
      "(mean:", mean(fraud_amt), ")\n")
  cat("Legitimate transactions:", legit_ci[1], "-", legit_ci[2], 
      "(mean:", mean(legit_amt), ")\n")
}

# Hypothesis Test 2: Chi-square test for ProductCD
if("ProductCD" %in% colnames(df)) {
  cat("\n", rep("=", 80), "\n", sep = "")
  cat("Chi-square Test: ProductCD and Fraud\n")
  cat(rep("=", 80), "\n", sep = "")
  
  contingency_table <- table(df$ProductCD, df$isFraud)
  cat("\nContingency Table:\n")
  print(contingency_table)
  
  chi_test <- chisq.test(contingency_table)
  cat("\nChi-square Test Results:\n")
  print(chi_test)
  
  if(chi_test$p.value < 0.05) {
    cat("Result: REJECT H0 - ProductCD is significantly associated with fraud\n")
  } else {
    cat("Result: FAIL TO REJECT H0 - No significant association\n")
  }
  
  # Visualization
  contingency_pct <- prop.table(contingency_table, margin = 1) * 100
  png(file.path(output_dir, "chi_square_productcd_r.png"), 
      width = 1200, height = 800, res = 300)
  barplot(t(contingency_pct), main = "Fraud Rate by ProductCD", 
          xlab = "ProductCD", ylab = "Percentage (%)",
          legend.text = c("Legitimate", "Fraud"), 
          args.legend = list(x = "topright"))
  dev.off()
  cat("\nFigure saved to", file.path(output_dir, "chi_square_productcd_r.png"), "\n")
}

# Hypothesis Test 3: Chi-square test for card type
if("card4" %in% colnames(df)) {
  cat("\n", rep("=", 80), "\n", sep = "")
  cat("Chi-square Test: Card Type (card4) and Fraud\n")
  cat(rep("=", 80), "\n", sep = "")
  
  contingency_table <- table(df$card4, df$isFraud)
  cat("\nContingency Table:\n")
  print(contingency_table)
  
  chi_test <- chisq.test(contingency_table)
  cat("\nChi-square Test Results:\n")
  print(chi_test)
  
  if(chi_test$p.value < 0.05) {
    cat("Result: REJECT H0 - Card type is significantly associated with fraud\n")
  } else {
    cat("Result: FAIL TO REJECT H0 - No significant association\n")
  }
}

# ============================================================================
# 3. EXPLORATORY STATISTICS
# ============================================================================
cat("\n", rep("=", 80), "\n", sep = "")
cat("EXPLORATORY STATISTICS\n")
cat(rep("=", 80), "\n", sep = "")

# Correlation analysis
cat("\nCorrelation Analysis\n")
key_features <- c("TransactionAmt", "card1", "card2", "card3", "card5", "isFraud")
key_features <- key_features[key_features %in% colnames(df)]

if(length(key_features) > 1) {
  # Calculate correlation matrix
  corr_matrix <- cor(df[key_features], use = "complete.obs")
  
  # Visualization
  png(file.path(output_dir, "correlation_matrix_statistical_r.png"), 
      width = 1200, height = 1000, res = 300)
  corrplot(corr_matrix, method = "color", type = "upper", 
           order = "hclust", tl.cex = 0.8, tl.col = "black",
           addCoef.col = "black", number.cex = 0.7)
  dev.off()
  cat("\nFigure saved to", file.path(output_dir, "correlation_matrix_statistical_r.png"), "\n")
  
  # Correlation with fraud
  if("isFraud" %in% key_features) {
    fraud_corr <- corr_matrix[, "isFraud"]
    fraud_corr <- sort(fraud_corr, decreasing = TRUE)
    cat("\nFeatures most correlated with Fraud:\n")
    print(fraud_corr)
  }
}

# Statistical summary by fraud status
cat("\n", rep("=", 80), "\n", sep = "")
cat("Statistical Summary by Fraud Status\n")
cat(rep("=", 80), "\n", sep = "")

analysis_features <- c("TransactionAmt", "card1", "card2", "card3", "card5")
analysis_features <- analysis_features[analysis_features %in% colnames(df)]

if(length(analysis_features) > 0) {
  summary_stats <- df %>%
    group_by(isFraud) %>%
    summarise(across(all_of(analysis_features), 
                     list(mean = ~mean(.x, na.rm = TRUE),
                          median = ~median(.x, na.rm = TRUE),
                          sd = ~sd(.x, na.rm = TRUE)),
                     .names = "{.col}_{.fn}"))
  cat("\nSummary Statistics by Fraud Status:\n")
  print(summary_stats)
  
  # Perform statistical tests for each feature
  cat("\nStatistical Tests for Each Feature:\n")
  test_results <- data.frame(
    Feature = character(),
    Test = character(),
    Statistic = numeric(),
    p.value = numeric(),
    Significant = character(),
    stringsAsFactors = FALSE
  )
  
  for(feature in analysis_features) {
    fraud_data <- df[[feature]][df$isFraud == 1 & !is.na(df[[feature]])]
    legit_data <- df[[feature]][df$isFraud == 0 & !is.na(df[[feature]])]
    
    if(length(fraud_data) > 0 && length(legit_data) > 0) {
      # Mann-Whitney U test
      test_result <- wilcox.test(fraud_data, legit_data, alternative = "two.sided")
      test_results <- rbind(test_results, data.frame(
        Feature = feature,
        Test = "Mann-Whitney U",
        Statistic = test_result$statistic,
        p.value = test_result$p.value,
        Significant = ifelse(test_result$p.value < 0.05, "Yes", "No"),
        stringsAsFactors = FALSE
      ))
    }
  }
  
  print(test_results)
}

# Summary
cat("\n", rep("=", 80), "\n", sep = "")
cat("STATISTICAL ANALYSIS SUMMARY\n")
cat(rep("=", 80), "\n", sep = "")
cat("\n1. Descriptive Statistics:\n")
cat("   - Calculated mean, median, std, skewness, and kurtosis for key features\n")
cat("   - Compared statistics between fraud and legitimate transactions\n")
cat("\n2. Inferential Statistics:\n")
cat("   - Performed hypothesis tests to compare fraud and legitimate transactions\n")
cat("   - Used non-parametric tests (Mann-Whitney U) due to non-normal distributions\n")
cat("   - Calculated confidence intervals for key metrics\n")
cat("   - Performed chi-square tests for categorical variables\n")
cat("\n3. Exploratory Statistics:\n")
cat("   - Analyzed correlations between features and fraud\n")
cat("   - Identified statistically significant relationships\n")
cat("   - Explored feature distributions by fraud status\n")
cat("\nStatistical Analysis Complete! Check outputs/figures/ for visualizations.\n")
cat(rep("=", 80), "\n", sep = "")

