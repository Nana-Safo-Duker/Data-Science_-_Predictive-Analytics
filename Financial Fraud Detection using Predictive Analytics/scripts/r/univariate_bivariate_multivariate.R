# Univariate, Bivariate, and Multivariate Analysis Script (R)

# Load libraries
library(tidyverse)
library(ggplot2)
library(corrplot)

# Load data
df <- read.csv("data/fraud_data.csv", stringsAsFactors = FALSE)
cat("Data loaded:", dim(df), "\n")

# Univariate analysis
if("TransactionAmt" %in% colnames(df)) {
  cat("Univariate Analysis: TransactionAmt\n")
  cat("Mean:", mean(df$TransactionAmt, na.rm = TRUE), "\n")
  cat("Median:", median(df$TransactionAmt, na.rm = TRUE), "\n")
  cat("Std:", sd(df$TransactionAmt, na.rm = TRUE), "\n")
}

# Bivariate analysis
if("TransactionAmt" %in% colnames(df)) {
  corr <- cor(df$TransactionAmt, df$isFraud, use = "complete.obs")
  cat("Correlation between TransactionAmt and isFraud:", corr, "\n")
}

# Multivariate analysis
key_features <- c("TransactionAmt", "card1", "card2", "card3", "card5", "isFraud")
key_features <- key_features[key_features %in% colnames(df)]

if(length(key_features) > 1) {
  corr_matrix <- cor(df[key_features], use = "complete.obs")
  png("outputs/figures/correlation_matrix_r.png", width = 800, height = 600)
  corrplot(corr_matrix, method = "color", type = "upper")
  dev.off()
  cat("Multivariate analysis complete!\n")
}

cat("Analysis Complete!\n")

