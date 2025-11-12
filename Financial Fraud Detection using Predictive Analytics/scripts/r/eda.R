# Exploratory Data Analysis Script (R)

# Load libraries
library(tidyverse)
library(ggplot2)

# Load data
df <- read.csv("data/fraud_data.csv", stringsAsFactors = FALSE)
cat("Data loaded:", dim(df), "\n")

# Target variable distribution
target_counts <- table(df$isFraud)
cat("Target Distribution:\n")
print(target_counts)

# Missing values
missing_data <- df %>%
  summarise_all(~sum(is.na(.))) %>%
  gather(key = "Column", value = "Missing_Count") %>%
  filter(Missing_Count > 0) %>%
  arrange(desc(Missing_Count))

cat("\nMissing Values:\n")
print(head(missing_data, 20))

# Transaction Amount Analysis
if("TransactionAmt" %in% colnames(df)) {
  cat("\nTransaction Amount Statistics:\n")
  cat("Mean:", mean(df$TransactionAmt, na.rm = TRUE), "\n")
  cat("Median:", median(df$TransactionAmt, na.rm = TRUE), "\n")
  cat("Std:", sd(df$TransactionAmt, na.rm = TRUE), "\n")
  
  # Save plot
  png("outputs/figures/transaction_amount_r.png", width = 800, height = 600)
  hist(df$TransactionAmt, main = "Transaction Amount Distribution", 
       xlab = "Transaction Amount", col = "steelblue")
  dev.off()
}

cat("\nEDA Complete!\n")

